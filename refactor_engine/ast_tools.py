"""Python source/AST tools: parsing, DFS, metrics, visualisation and unparsing."""

from __future__ import annotations

import ast
import io
import textwrap
import tokenize
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

# Parsing: source text -> AST.

@dataclass
class SyntaxErrorInfo:
    message: str
    line: int | None
    column: int | None
    text: str | None

    def pretty(self) -> str:
        where = f"line {self.line}" if self.line else "unknown line"
        out = f"SyntaxError: {self.message} ({where})"
        if self.text:
            out += "\n    " + self.text.rstrip("\n")
            if self.column:
                out += "\n    " + " " * (self.column - 1) + "^"
        return out


@dataclass
class ParseResult:
    source: str
    tree: ast.Module | None
    error: SyntaxErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.tree is not None


def normalize_source(source: str | bytes) -> str:
    """Decode, unify newlines, strip a BOM and remove common indentation.

    Removing common indentation lets users paste a snippet copied from the
    middle of a class or function without getting an IndentationError.
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8", errors="replace")
    source = source.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    source = source.expandtabs(4)
    source = textwrap.dedent(source)
    lines = [line.rstrip() for line in source.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    return "\n".join(lines) + "\n" if lines else ""


def parse_source(source: str, filename: str = "<input>") -> ParseResult:
    """Parse source code into an AST, capturing syntax errors."""
    try:
        tree = ast.parse(source, filename=filename, type_comments=False)
    except SyntaxError as exc:
        return ParseResult(
            source=source,
            tree=None,
            error=SyntaxErrorInfo(exc.msg, exc.lineno, exc.offset, exc.text),
        )
    except (ValueError, RecursionError) as exc:  # e.g. null bytes, absurd nesting
        return ParseResult(source=source, tree=None, error=SyntaxErrorInfo(str(exc), None, None, None))
    return ParseResult(source=source, tree=tree)


# Nodes that carry no structural information of their own: expression
# contexts and operator singletons. They can be hidden in visualisations.
CONTEXT_NODES = (ast.Load, ast.Store, ast.Del)
OPERATOR_NODES = (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)

OP_SYMBOLS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.LShift: "<<", ast.RShift: ">>",
    ast.BitOr: "|", ast.BitXor: "^", ast.BitAnd: "&", ast.MatMult: "@",
    ast.And: "and", ast.Or: "or", ast.Not: "not", ast.USub: "-", ast.UAdd: "+",
    ast.Invert: "~", ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not", ast.In: "in",
    ast.NotIn: "not in",
}

SCOPE_NODES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class TraversalEntry:
    order: int
    node: ast.AST
    depth: int
    parent: ast.AST | None
    field: str | None
    index: int | None


def iter_child_slots(node: ast.AST) -> Iterator[tuple[str, int | None, ast.AST]]:
    """Yield ``(field, index, child)`` for every AST child, left to right."""
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, ast.AST):
                    yield field, i, item
        elif isinstance(value, ast.AST):
            yield field, None, value


def dfs(tree: ast.AST, include_trivial: bool = True) -> list[TraversalEntry]:
    """Iterative pre-order Depth-First Search over the AST.

    An explicit stack is used (instead of recursion) so that very deep trees
    cannot overflow the interpreter stack. Children are pushed in reverse so
    they are visited left-to-right, i.e. in source order.
    """
    order: list[TraversalEntry] = []
    stack: list[tuple[ast.AST, int, ast.AST | None, str | None, int | None]] = [(tree, 0, None, None, None)]
    while stack:
        node, depth, parent, field, index = stack.pop()
        if not include_trivial and isinstance(node, CONTEXT_NODES + OPERATOR_NODES):
            continue
        order.append(TraversalEntry(len(order), node, depth, parent, field, index))
        for child_field, child_index, child in reversed(list(iter_child_slots(node))):
            stack.append((child, depth + 1, node, child_field, child_index))
    return order


def parent_map(tree: ast.AST) -> dict[int, tuple[ast.AST, str, int | None]]:
    """Map ``id(child) -> (parent, field, index)``."""
    return {id(e.node): (e.parent, e.field, e.index) for e in dfs(tree) if e.parent is not None}


def short_repr(value: object, limit: int = 24) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def node_label(node: ast.AST) -> str:
    """Human readable one-line label for a node (used in tables and graphs)."""
    name = type(node).__name__
    if isinstance(node, ast.Name):
        return f"Name: {node.id}"
    if isinstance(node, ast.Constant):
        return f"Constant: {short_repr(node.value)}"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return f"{name}: {node.name}"
    if isinstance(node, ast.arg):
        return f"arg: {node.arg}"
    if isinstance(node, ast.Attribute):
        return f"Attribute: .{node.attr}"
    if isinstance(node, ast.alias):
        return f"alias: {node.name}" + (f" as {node.asname}" if node.asname else "")
    if isinstance(node, ast.keyword):
        return f"keyword: {node.arg or '**'}"
    if isinstance(node, (ast.BinOp, ast.AugAssign, ast.UnaryOp, ast.BoolOp)):
        return f"{name} ({OP_SYMBOLS.get(type(node.op), '?')})"
    if isinstance(node, ast.Compare):
        return "Compare (" + " ".join(OP_SYMBOLS.get(type(o), "?") for o in node.ops) + ")"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        mod = getattr(node, "module", None)
        return f"{name}" + (f": {mod}" if mod else "")
    if isinstance(node, ast.ExceptHandler) and node.name:
        return f"ExceptHandler: {node.name}"
    if isinstance(node, OPERATOR_NODES) and type(node) in OP_SYMBOLS:
        return f"{name} ({OP_SYMBOLS[type(node)]})"
    return name


def node_category(node: ast.AST) -> str:
    if isinstance(node, (ast.Module,)):
        return "module"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.arguments, ast.arg)):
        return "definition"
    if isinstance(node, ast.stmt):
        return "statement"
    if isinstance(node, ast.Constant):
        return "literal"
    if isinstance(node, ast.Name):
        return "name"
    if isinstance(node, ast.expr):
        return "expression"
    if isinstance(node, CONTEXT_NODES + OPERATOR_NODES):
        return "trivial"
    return "other"


def count_nodes(node: ast.AST | list, include_trivial: bool = False) -> int:
    nodes = node if isinstance(node, list) else [node]
    return sum(include_trivial or not isinstance(sub, CONTEXT_NODES + OPERATOR_NODES)
               for root in nodes for sub in ast.walk(root))


def expr_depth(node: ast.AST | list) -> int:
    """Height of the subtree, ignoring context/operator singletons."""
    if isinstance(node, list):
        return max((expr_depth(n) for n in node), default=0)
    children = [c for _, _, c in iter_child_slots(node) if not isinstance(c, CONTEXT_NODES + OPERATOR_NODES)]
    return 1 + max((expr_depth(c) for c in children), default=0)


# Code metrics (used to show the effect of refactoring / optimisation)

DECISION_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler, ast.Assert, ast.match_case)


def cyclomatic_complexity(tree: ast.AST) -> int:
    """McCabe complexity approximation: 1 + number of decision points."""
    score = 1
    for node in ast.walk(tree):
        if isinstance(node, DECISION_NODES):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            score += 1 + len(node.ifs)
    return score


def max_nesting(tree: ast.AST) -> int:
    """Deepest nesting of compound statements (if/for/while/with/try/def)."""
    compound = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try,
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Match)
    if hasattr(ast, "TryStar"):
        compound = compound + (ast.TryStar,)
    best = 0
    stack = [(tree, 0)]
    while stack:
        node, level = stack.pop()
        best = max(best, level)
        for _, _, child in iter_child_slots(node):
            stack.append((child, level + 1 if isinstance(child, compound) else level))
    return best


def code_metrics(source: str, tree: ast.AST | None) -> dict[str, int]:
    lines = source.splitlines()
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    metrics = {
        "lines": len(lines),
        "code_lines": len(code_lines),
        "characters": len(source),
    }
    if tree is not None:
        metrics.update(
            ast_nodes=count_nodes(tree, include_trivial=True),
            statements=sum(isinstance(n, ast.stmt) for n in ast.walk(tree)),
            functions=sum(isinstance(n, FUNCTION_NODES) for n in ast.walk(tree)),
            complexity=cyclomatic_complexity(tree),
            max_nesting=max_nesting(tree),
            ast_depth=expr_depth(tree),
        )
    return metrics


# Visualisation

def scope_nodes(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Top-level choices for the AST viewer: whole module and each def/class."""
    choices: list[tuple[str, ast.AST]] = [("Whole module", tree)]
    for entry in dfs(tree):
        node = entry.node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            choices.append((f"{kind} {node.name} (line {node.lineno})", node))
    return choices


_COLORS = {
    "module": ("#1f2937", "#ffffff"),
    "definition": ("#7c3aed", "#ffffff"),
    "statement": ("#2563eb", "#ffffff"),
    "expression": ("#d1fae5", "#065f46"),
    "name": ("#ecfccb", "#3f6212"),
    "literal": ("#fef3c7", "#92400e"),
    "trivial": ("#f3f4f6", "#6b7280"),
    "other": ("#e5e7eb", "#374151"),
}


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def to_graphviz(
    root: ast.AST,
    highlight: set[int] | None = None,
    include_trivial: bool = False,
    max_nodes: int = 350,
    show_fields: bool = True,
) -> tuple[str, int, bool]:
    """Render the (sub)tree as a Graphviz DOT string.

    Returns ``(dot, rendered_nodes, truncated)``. Nodes whose ``id`` is in
    ``highlight`` are drawn in red - these are nodes involved in a detected
    refactoring opportunity.
    """
    highlight = highlight or set()
    lines = [
        "digraph AST {",
        '  graph [rankdir=TB, bgcolor="transparent", nodesep=0.25, ranksep=0.35];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, height=0.3];',
        '  edge [fontname="Helvetica", fontsize=8, color="#9ca3af", fontcolor="#6b7280", arrowsize=0.6];',
    ]
    ids: dict[int, str] = {}
    truncated = False
    for entry in dfs(root, include_trivial=include_trivial):
        if len(ids) >= max_nodes:
            truncated = True
            break
        node = entry.node
        if entry.parent is not None and id(entry.parent) not in ids:
            continue
        nid = f"n{len(ids)}"
        ids[id(node)] = nid
        fill, font = _COLORS[node_category(node)]
        extra = ""
        if id(node) in highlight:
            fill, font = "#fee2e2", "#991b1b"
            extra = ', color="#dc2626", penwidth=2.2'
        label = _dot_escape(node_label(node))
        if isinstance(node, ast.stmt) and hasattr(node, "lineno"):
            label += f"\\nline {node.lineno}"
        lines.append(f'  {nid} [label="{label}", fillcolor="{fill}", fontcolor="{font}"{extra}];')
        if entry.parent is not None:
            edge_label = ""
            if show_fields and entry.field:
                edge_label = entry.field + (f"[{entry.index}]" if entry.index is not None else "")
            lines.append(f'  {ids[id(entry.parent)]} -> {nid} [label="{_dot_escape(edge_label)}"];')
    lines.append("}")
    return "\n".join(lines), len(ids), truncated


# Source generation, with the input's quote style.
_prefer_double: ContextVar[bool] = ContextVar("prefer_double_quotes", default=False)

_Base = getattr(ast, "_Unparser", None)


def detect_quote_preference(source: str) -> bool:
    """True if the source mostly uses double-quoted strings."""
    single = double = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.STRING:
                body = tok.string.lstrip("rRbBuUfF")
                if body.startswith(('"""', "'''")):
                    continue
                double += body.startswith('"')
                single += body.startswith("'")
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return double > single


def set_quote_preference(prefer_double: bool):
    return _prefer_double.set(prefer_double)


if _Base is not None:
    class _PrettyUnparser(_Base):
        def visit_Call(self, node):
            if len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.GeneratorExp):
                self.set_precedence(ast._Precedence.ATOM, node.func)
                self.traverse(node.func)
                gen = node.args[0]
                self.write("(")
                self.traverse(gen.elt)
                for comp in gen.generators:
                    self.traverse(comp)
                self.write(")")
                return
            super().visit_Call(node)

        def _write_constant(self, value):
            if isinstance(value, str) and _prefer_double.get() and '"' not in value:
                literal = repr(value)
                if literal.startswith("'"):  # repr only picks ' when the text has no '
                    self.write('"' + literal[1:-1] + '"')
                    return
            super()._write_constant(value)


def unparse(node: ast.AST) -> str:
    """Generate source code for ``node`` (also accepts nodes without positions)."""
    node = ast.fix_missing_locations(node)
    if _Base is None:
        return ast.unparse(node)
    try:
        return _PrettyUnparser().visit(node)
    except Exception:
        return ast.unparse(node)
