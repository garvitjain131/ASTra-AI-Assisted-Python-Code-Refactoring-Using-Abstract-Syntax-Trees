"""Build the training dataset and train the refactoring ranker.

Usage:  python scripts/train_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refactor_engine.ml_model import DATASET_PATH, METRICS_PATH, MODEL_PATH, train  # noqa: E402

if __name__ == "__main__":
    print("Generating dataset and comparing models (5-fold CV)...")
    result = train(save=True, verbose=True)
    m = result["metrics"]
    df = result["dataset"]
    print(f"\nDataset: {len(df)} rows, {df['label'].mean():.1%} positive, {df['rule_id'].nunique()} rule types")
    print(f"Selected model: {m['model']}")
    print(f"Hold-out test  accuracy={m['accuracy']:.3f}  precision={m['precision']:.3f}  "
          f"recall={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("Top features:")
    for f in m["feature_importance"][:8]:
        print(f"  {f['feature']:<22} {f['importance']:.3f}")
    print(f"\nSaved model   -> {MODEL_PATH}\nSaved metrics -> {METRICS_PATH}\nSaved dataset -> {DATASET_PATH}")
