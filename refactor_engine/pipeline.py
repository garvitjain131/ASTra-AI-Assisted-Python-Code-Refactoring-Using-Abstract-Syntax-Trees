"""Parse -> analyse -> rank -> rewrite -> generate, repeating until no approved change remains."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field

from .analysis import Analyzer, extract_features
from .ast_tools import (ParseResult, code_metrics, detect_quote_preference, dfs,
                        normalize_source, parse_source, set_quote_preference)
from .codegen import apply_and_generate, unified_diff
from .ml_model import RefactoringRanker
from .rules import get_rules

MAX_STEPS = 200

_RANKER: RefactoringRanker | None = None


def get_ranker() -> RefactoringRanker:
    global _RANKER
    if _RANKER is None:
        _RANKER = RefactoringRanker()
    return _RANKER


def safety_label(risk: float) -> str:
    if risk <= 0.0:
        return "Safe"
    if risk < 0.25:
        return "Low risk"
    if risk < 0.5:
        return "Review"
    return "Risky"


@dataclass
class Suggestion:
    uid: str
    rule_id: str
    rule_name: str
    category: str
    line: int
    end_line: int
    title: str
    explanation: str
    before_code: str
    after_code: str
    risk: float
    safety: str
    score: float
    recommended: bool
    will_apply: bool
    features: dict
    highlight_ids: set = field(default_factory=set, repr=False)

    @property
    def key(self) -> tuple[str, str]:
        return (self.rule_id, self.before_code)


@dataclass
class AppliedStep:
    step: int
    rule_id: str
    rule_name: str
    title: str
    line: int
    before_code: str
    after_code: str
    score: float
    method: str


@dataclass
class PipelineResult:
    source: str
    parse: ParseResult
    suggestions: list[Suggestion] = field(default_factory=list)
    refactored_source: str = ""
    refactored_tree: ast.Module | None = None
    steps: list[AppliedStep] = field(default_factory=list)
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    model_info: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.parse.ok

    @property
    def changed(self) -> bool:
        return self.ok and self.refactored_source != self.source

    def diff(self) -> str:
        return unified_diff(self.source, self.refactored_source)


class _ScoreCache:
    def __init__(self, ranker: RefactoringRanker):
        self.ranker, self.cache = ranker, {}

    def __call__(self, rows: list[dict]) -> list[float]:
        keys = [tuple(sorted(r.items())) for r in rows]
        missing = [(k, r) for k, r in zip(keys, rows) if k not in self.cache]
        if missing:
            for (k, _), s in zip(missing, self.ranker.score([r for _, r in missing])):
                self.cache[k] = s
        return [self.cache[k] for k in keys]


def _decide(match_key, score: float, threshold: float, apply_mode: str, overrides: dict) -> bool:
    if match_key in overrides:
        return bool(overrides[match_key])
    if apply_mode == "all":
        return True
    if apply_mode == "none":
        return False
    return score >= threshold


def _rank_matches(analyzer, scorer, tree, source, timings=None):
    """One shared analysis/ranking path for suggestions and follow-up edits."""
    started = time.perf_counter()
    matches = analyzer.analyze(tree, source)
    analysed = time.perf_counter()
    rows = [extract_features(match) for match in matches]
    ranked = sorted(zip(matches, rows, scorer(rows)), key=lambda item: (-item[2], item[0].line))
    if timings is not None:
        timings.update(analysis=analysed - started, ml=time.perf_counter() - analysed)
    return ranked


def _refactor(result, analyzer, scorer, decide, preserve):
    """Apply the best approved match, then analyse the new program again."""
    current = result.source
    tree = parse_source(current).tree  # keep the original tree intact for the AST viewer
    seen, failed = {current}, set()
    for _ in range(MAX_STEPS):
        candidates = [(m, score, m.choice_key)
                      for m, _, score in _rank_matches(analyzer, scorer, tree, current)
                      if m.choice_key not in failed and decide(m.choice_key, score)]
        if not candidates:
            break
        match, score, key = candidates[0]
        step_info = (match.rule.rule_id, match.rule.name, match.title, match.line,
                     match.before_code(), match.after_code())
        try:
            new_source, method = apply_and_generate(current, tree, match, preserve)
            new_parse = parse_source(new_source)
        except Exception as exc:  # a rule bug must not crash the UI
            result.warnings.append(f"{match.rule.rule_id} skipped at line {match.line}: {exc}")
            new_parse = None
        if new_parse is None or not new_parse.ok or new_source in seen:
            failed.add(key)
            tree = parse_source(current).tree  # discard a failed edit's mutated tree
            continue
        if method == "regenerate" and preserve and not any(w.startswith("Some changes") for w in result.warnings):
            result.warnings.append("Some changes could not be spliced into the original text, so the code was "
                                   "regenerated from the AST; comments may have been removed.")
        result.steps.append(AppliedStep(len(result.steps) + 1, *step_info, score, method))
        current, tree = new_source, new_parse.tree
        seen.add(current)
    result.refactored_source, result.refactored_tree = current, tree


def run_pipeline(
    source: str,
    rule_ids=None,
    threshold: float = 0.5,
    apply_mode: str = "recommended",
    preserve_formatting: bool = True,
    overrides: dict | None = None,
    ranker: RefactoringRanker | None = None,
) -> PipelineResult:
    """Analyse and refactor ``source``.

    apply_mode: ``"recommended"`` (ML score >= threshold), ``"all"`` or ``"none"``.
    overrides:  ``{(rule_id, before_code): bool}`` to force a decision for a suggestion.
    """
    overrides = overrides or {}
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # 1. Parse the source into our intermediate representation: the AST.
    source = normalize_source(source)
    parsed = parse_source(source)
    set_quote_preference(detect_quote_preference(source))
    timings["parse"] = time.perf_counter() - t0
    result = PipelineResult(source=source, parse=parsed, refactored_source=source)
    if not parsed.ok:
        return result

    ranker = ranker or get_ranker()
    scorer = _ScoreCache(ranker)
    analyzer = Analyzer(get_rules(rule_ids))
    result.model_info = {"available": ranker.available, "error": ranker.error, **ranker.metrics}
    if not ranker.available:
        result.warnings.append("ML model unavailable - using the rule-risk heuristic for ranking. "
                               f"({ranker.error})")

    def decide(key, score):
        return _decide(key, score, threshold, apply_mode, overrides)

    # 2. Detect and rank opportunities; retain them for explanations and highlighting.
    for i, (m, feats, score) in enumerate(_rank_matches(analyzer, scorer, parsed.tree, source, timings)):
        key = m.choice_key
        result.suggestions.append(Suggestion(
            uid=f"S{i + 1}", rule_id=m.rule.rule_id, rule_name=m.rule.name, category=m.rule.category,
            line=m.line, end_line=m.end_line, title=m.title, explanation=m.explanation,
            before_code=key[1], after_code=m.after_code(), risk=m.risk, safety=safety_label(m.risk),
            score=score, recommended=score >= threshold,
            will_apply=decide(key, score),
            features=feats, highlight_ids={id(n) for root in m.original_nodes for n in ast.walk(root)},
        ))
    # 3. Transform the AST and generate verified source until no approved edit remains.
    t = time.perf_counter()
    _refactor(result, analyzer, scorer, decide, preserve_formatting)
    timings["refactor"] = time.perf_counter() - t

    # 4. Report the effect of the changes.
    result.metrics_before = code_metrics(source, parsed.tree)
    result.metrics_after = code_metrics(result.refactored_source, result.refactored_tree)
    if not preserve_formatting and any("#" in line for line in source.splitlines()):
        result.warnings.append("AST regeneration mode: comments are not part of the AST and are dropped.")
    timings["total"] = time.perf_counter() - t0
    result.timings = timings
    return result


def traversal_table(tree: ast.AST, include_trivial: bool = False) -> list[dict]:
    """DFS order of the AST, as rows for display."""
    from .ast_tools import node_label

    rows = []
    for e in dfs(tree, include_trivial=include_trivial):
        rows.append({
            "order": e.order,
            "depth": e.depth,
            "node": node_label(e.node),
            "type": type(e.node).__name__,
            "field": (e.field or "") + (f"[{e.index}]" if e.index is not None else ""),
            "line": getattr(e.node, "lineno", None),
        })
    return rows
