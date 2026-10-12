"""Train, compare, save and load classifiers that rank AST refactoring suggestions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .analysis import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODEL_PATH = MODEL_DIR / "refactoring_ranker.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DATASET_PATH = DATA_DIR / "training_data.csv"
MANUAL_LABELS_PATH = DATA_DIR / "manual_labels.csv"


def _make_pipeline(estimator):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer([
        ("rule", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([("features", pre), ("model", estimator)])


def candidate_models() -> dict:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
    }


def train(df: pd.DataFrame | None = None, save: bool = True, verbose: bool = False) -> dict:
    """Build the dataset (if needed), compare models, fit the best, save it."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    from .dataset import build_dataset

    started = time.time()
    if df is None:
        df = build_dataset()
        if MANUAL_LABELS_PATH.exists():
            manual = pd.read_csv(MANUAL_LABELS_PATH)
            df = pd.concat([df, manual[FEATURE_COLUMNS + ["label"]]], ignore_index=True)
    X, y = df[FEATURE_COLUMNS], df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = candidate_models()
    comparison = []
    for model_name, est in models.items():
        scores = cross_val_score(_make_pipeline(est), X_train, y_train, cv=cv, scoring="f1")
        comparison.append({"model": model_name, "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std())})
        if verbose:
            print(f"  {model_name:<20} CV F1 = {scores.mean():.3f} +/- {scores.std():.3f}")
    best_name = max(comparison, key=lambda r: r["cv_f1_mean"])["model"]
    pipeline = _make_pipeline(models[best_name]).fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    metrics = {
        "model": best_name,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "comparison": comparison,
        "feature_importance": _feature_importance(pipeline),
        "train_seconds": round(time.time() - started, 2),
    }
    # Refit on all data for the deployed model.
    pipeline = _make_pipeline(models[best_name]).fit(X, y)
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH, compress=3)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        df.to_csv(DATASET_PATH, index=False)
    return {"pipeline": pipeline, "metrics": metrics, "dataset": df}


def _feature_importance(pipeline) -> list[dict]:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["features"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return []
    pretty = [n.replace("rule__rule_id_", "rule=").replace("num__", "") for n in names]
    pairs = sorted(zip(pretty, values), key=lambda p: -p[1])
    return [{"feature": n, "importance": float(v)} for n, v in pairs]


class RefactoringRanker:
    """Loads (or trains on first use) the model and scores feature rows."""

    def __init__(self, model_path: Path = MODEL_PATH, auto_train: bool = True):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.metrics: dict = {}
        self.error: str | None = None
        try:
            if self.model_path.exists():
                self.pipeline = joblib.load(self.model_path)
                if METRICS_PATH.exists():
                    self.metrics = json.loads(METRICS_PATH.read_text())
            elif auto_train:
                result = train(save=True)
                self.pipeline, self.metrics = result["pipeline"], result["metrics"]
        except Exception as exc:  # sklearn missing / incompatible pickle -> heuristic fallback
            self.error = f"{type(exc).__name__}: {exc}"
            self.pipeline = None

    @property
    def available(self) -> bool:
        return self.pipeline is not None

    def score(self, rows: list[dict]) -> list[float]:
        if not rows:
            return []
        if self.pipeline is None:
            # Fallback when no model is available: safer rewrites rank higher.
            return [round(max(0.0, 1.0 - r["risk"] - 0.004 * max(0, r["max_line_len_after"] - 79)), 4) for r in rows]
        frame = pd.DataFrame(rows)[FEATURE_COLUMNS]
        proba = self.pipeline.predict_proba(frame)[:, 1]
        return [round(float(p), 4) for p in proba]

    def retrain(self) -> dict:
        result = train(save=True)
        self.pipeline, self.metrics, self.error = result["pipeline"], result["metrics"], None
        return self.metrics
