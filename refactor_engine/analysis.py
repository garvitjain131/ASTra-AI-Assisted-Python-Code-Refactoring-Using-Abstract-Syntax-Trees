"""Build scope information, detect rule matches, and extract their ML features."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .ast_tools import FUNCTION_NODES, SCOPE_NODES, TraversalEntry, count_nodes, dfs, expr_depth, unparse

LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)


# Scope / symbol information

@dataclass
class ScopeInfo:
    node: ast.AST
    parent: "ScopeInfo | None" = None
    loads: Counter = field(default_factory=Counter)
    stores: Counter = field(default_factory=Counter)
    declared: set = field(default_factory=set)       # global / nonlocal names
    closure_reads: set = field(default_factory=set)  # names read by nested scopes

    @property
    def is_function(self) -> bool:
        return isinstance(self.node, FUNCTION_NODES)

    @property
    def is_module(self) -> bool:
        return isinstance(self.node, ast.Module)

    def all_names(self) -> set[str]:
        return set(self.loads) | set(self.stores) | self.declared


def build_scopes(tree: ast.Module, order: list[TraversalEntry]) -> tuple[dict[int, ScopeInfo], dict[int, ScopeInfo]]:
    """Return ``(scope_by_scope_node_id, scope_of_node_id)``."""
    scopes: dict[int, ScopeInfo] = {id(tree): ScopeInfo(tree)}
    scope_of: dict[int, ScopeInfo] = {id(tree): scopes[id(tree)]}
    for entry in order[1:]:
        node, parent = entry.node, entry.parent
        parent_scope = scope_of[id(parent)]
        # Decorators, default values and annotations of a def are evaluated in
        # the *enclosing* scope; everything else under a scope node is inside it.
        if isinstance(parent, SCOPE_NODES) and entry.field not in ("decorator_list", "returns", "bases", "keywords"):
            if id(parent) not in scopes:
                enclosing = scope_of[id(parent)]
                scopes[id(parent)] = ScopeInfo(parent, parent=enclosing)
            parent_scope = scopes[id(parent)]
        scope_of[id(node)] = parent_scope

        scope = parent_scope
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                scope.loads[node.id] += 1
                outer = scope.parent
                while outer is not None:
                    outer.loads[node.id] += 1
                    outer.closure_reads.add(node.id)
                    outer = outer.parent
            else:
                scope.stores[node.id] += 1
        elif isinstance(node, ast.arg):
            scope.stores[node.arg] += 1
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            scope.declared.update(node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.stores[node.name] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                scope.stores[(alias.asname or alias.name).split(".")[0]] += 1
    return scopes, scope_of


# Matches produced by the rules

@dataclass
class StmtSpan:
    """Statements ``container[start:end]`` are replaced by new statements."""
    container: list
    start: int
    end: int

    def nodes(self) -> list[ast.stmt]:
        return self.container[self.start:self.end]


@dataclass
class ExprSlot:
    """The expression stored at ``parent.field[index]`` is replaced."""
    parent: ast.AST
    field: str
    index: int | None

    def node(self) -> ast.expr:
        value = getattr(self.parent, self.field)
        return value[self.index] if self.index is not None else value


Location = StmtSpan | ExprSlot


@dataclass
class Match:
    rule: "object"                  # refactor_engine.rules.Rule
    location: Location
    replacement: object             # list[ast.stmt] for StmtSpan, ast.expr for ExprSlot
    title: str
    explanation: str
    risk: float                     # 0 = provably safe ... 1 = likely changes behaviour
    ancestors: list[ast.stmt]       # enclosing statements, innermost first
    context: "NodeContext"
    extra: dict = field(default_factory=dict)

    @property
    def anchor(self) -> ast.AST:
        return self.original_nodes[0]

    @property
    def original_nodes(self) -> list[ast.AST]:
        if isinstance(self.location, StmtSpan):
            return self.location.nodes()
        return [self.location.node()]

    @property
    def line(self) -> int:
        return getattr(self.original_nodes[0], "lineno", 0)

    @property
    def end_line(self) -> int:
        return getattr(self.original_nodes[-1], "end_lineno", self.line)

    def replacement_nodes(self) -> list[ast.AST]:
        return list(self.replacement) if isinstance(self.replacement, list) else [self.replacement]

    def before_code(self) -> str:
        return "\n".join(unparse(n) for n in self.original_nodes)

    def after_code(self) -> str:
        nodes = self.replacement_nodes()
        if not nodes:
            return "# (removed)"
        return "\n".join(unparse(n) for n in nodes)

    def key(self) -> tuple:
        return (self.rule.rule_id, self.line, self.before_code())

    @property
    def choice_key(self) -> tuple[str, str]:
        """Stable across line shifts while follow-up edits are applied."""
        return (self.rule.rule_id, self.before_code())


# Context handed to rules

@dataclass
class AnalysisState:
    tree: ast.Module
    source: str
    order: list[TraversalEntry]
    parents: dict[int, TraversalEntry]
    scopes: dict[int, ScopeInfo]
    scope_of: dict[int, ScopeInfo]
    module_loads: set[str]
    dunder_all: set[str]


@dataclass
class NodeContext:
    state: AnalysisState
    entry: TraversalEntry

    @property
    def node(self) -> ast.AST:
        return self.entry.node

    @property
    def scope(self) -> ScopeInfo:
        return self.state.scope_of[id(self.entry.node)]

    def parent_entry(self, node: ast.AST | None = None) -> TraversalEntry | None:
        node = node if node is not None else self.entry.node
        return self.state.parents.get(id(node))

    def ancestors(self, node: ast.AST | None = None) -> list[ast.AST]:
        out, cur = [], node if node is not None else self.entry.node
        while True:
            p = self.state.parents.get(id(cur))
            if p is None or p.parent is None:
                return out
            out.append(p.parent)
            cur = p.parent

    def ancestor_statements(self, node: ast.AST | None = None) -> list[ast.stmt]:
        return [a for a in self.ancestors(node) if isinstance(a, ast.stmt)]

    def block_depth(self, node: ast.AST | None = None) -> int:
        """Indentation level of ``node`` (0 = module level)."""
        return sum(isinstance(a, (ast.stmt, ast.ExceptHandler, ast.match_case)) and hasattr(a, "body")
                   for a in self.ancestors(node))

    def loop_depth(self, node: ast.AST | None = None) -> int:
        return sum(isinstance(a, LOOP_NODES) for a in self.ancestors(node))

    def scope_for(self, node: ast.AST) -> ScopeInfo:
        return self.state.scope_of[id(node)]

    def in_bool_context(self, node: ast.AST | None = None) -> bool:
        """True if the value of ``node`` is only used for its truthiness."""
        cur = node if node is not None else self.entry.node
        while True:
            p = self.state.parents.get(id(cur))
            if p is None or p.parent is None:
                return False
            parent, fld = p.parent, p.field
            if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)) and fld == "test":
                return True
            if isinstance(parent, ast.comprehension) and fld == "ifs":
                return True
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
                return True
            if isinstance(parent, ast.BoolOp):
                cur = parent  # truthiness only matters if the BoolOp itself is a condition
                continue
            return False

    def lines(self) -> list[str]:
        return self.state.source.splitlines()


class Analyzer:
    """Runs all enabled rules over the AST in a single DFS pass."""

    def __init__(self, rules: Iterable):
        self.rules = list(rules)

    @staticmethod
    def prepare(tree: ast.Module, source: str) -> AnalysisState:
        order = dfs(tree)
        parents = {id(e.node): e for e in order if e.parent is not None}
        scopes, scope_of = build_scopes(tree, order)
        module_loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        module_loads |= {_attr_root(n) for n in ast.walk(tree) if isinstance(n, ast.Attribute)} - {None}
        dunder_all: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    dunder_all |= {e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        return AnalysisState(tree, source, order, parents, scopes, scope_of, module_loads, dunder_all)

    def analyze(self, tree: ast.Module, source: str) -> list[Match]:
        state = self.prepare(tree, source)
        matches: list[Match] = []
        for entry in state.order:
            ctx = NodeContext(state, entry)
            node = entry.node
            for rule in self.rules:
                matches.extend(rule.match_node(node, ctx) or ())
            for fld, value in ast.iter_fields(node):
                if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                    for rule in self.rules:
                        matches.extend(rule.match_block(value, node, fld, ctx) or ())
        return matches


def _attr_root(node: ast.Attribute) -> str | None:
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return cur.id if isinstance(cur, ast.Name) else None


# Measurements passed to the ML ranker.
CATEGORICAL_FEATURES = ["rule_id"]
NUMERIC_FEATURES = ["risk", "nodes_before", "nodes_after", "node_reduction", "lines_before", "lines_after",
                    "line_reduction", "max_line_len_after", "replacement_len", "expr_depth_after",
                    "block_depth", "loop_depth", "calls_in_region", "bool_ops_after", "comp_generators",
                    "comp_has_if", "name_len", "in_function"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def extract_features(match: Match) -> dict:
    ctx = match.context
    before_nodes = match.original_nodes
    after_nodes = match.replacement_nodes()
    before_code = match.before_code()
    after_code = "" if not after_nodes else match.after_code()
    after_lines = after_code.splitlines() or [""]
    replacement_len = max(len(line) for line in after_lines)
    depth = ctx.block_depth(match.anchor)

    max_len = depth * 4 + replacement_len
    if isinstance(match.location, ExprSlot):
        node = match.location.node()
        source_lines = ctx.lines()
        if getattr(node, "lineno", None) == getattr(node, "end_lineno", -1) and node.lineno <= len(source_lines):
            line = source_lines[node.lineno - 1]
            max_len = len(line) - (node.end_col_offset - node.col_offset) + replacement_len

    nodes_before = count_nodes(before_nodes)
    nodes_after = count_nodes(after_nodes) if after_nodes else 0
    lines_before = len(before_code.splitlines())
    lines_after = len(after_code.splitlines()) if after_nodes else 0
    return {
        "rule_id": match.rule.rule_id,
        "risk": float(match.risk),
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "node_reduction": nodes_before - nodes_after,
        "lines_before": lines_before,
        "lines_after": lines_after,
        "line_reduction": lines_before - lines_after,
        "max_line_len_after": max_len,
        "replacement_len": replacement_len,
        "expr_depth_after": expr_depth(after_nodes) if after_nodes else 0,
        "block_depth": depth,
        "loop_depth": ctx.loop_depth(match.anchor),
        "calls_in_region": sum(isinstance(n, ast.Call) for root in before_nodes for n in ast.walk(root)),
        "bool_ops_after": sum(isinstance(n, (ast.BoolOp, ast.Compare, ast.IfExp))
                              for root in after_nodes for n in ast.walk(root)),
        "comp_generators": int(match.extra.get("comp_generators", 0)),
        "comp_has_if": int(match.extra.get("comp_has_if", 0)),
        "name_len": int(match.extra.get("name_len", 0)),
        "in_function": int(ctx.scope_for(match.anchor).is_function),
    }
