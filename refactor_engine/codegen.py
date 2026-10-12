"""Apply AST replacements and generate source, verifying each text edit by reparsing."""

from __future__ import annotations

import ast
import difflib
import io
import tokenize
from dataclasses import dataclass
from typing import Callable

from .analysis import ExprSlot, Match, StmtSpan
from .ast_tools import unparse


def apply_match(match: Match) -> None:
    """Rule-based AST transformation: splice the replacement into the tree."""
    loc = match.location
    if isinstance(loc, StmtSpan):
        new = list(match.replacement)
        if not new and loc.start == 0 and loc.end == len(loc.container):
            new = [ast.Pass()]  # a block can never be empty
        loc.container[loc.start:loc.end] = new
    else:
        if loc.index is None:
            setattr(loc.parent, loc.field, match.replacement)
        else:
            getattr(loc.parent, loc.field)[loc.index] = match.replacement


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _blank_lines_before_nested_defs(code: str) -> str:
    """PEP 8 layout inside a top-level statement: one blank line before nested defs."""
    out: list[str] = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        nested_def = line[:1].isspace() and stripped.startswith(("def ", "async def ", "class ", "@"))
        if nested_def and out and out[-1].strip() and not out[-1].lstrip().startswith("@") \
                and not out[-1].rstrip().endswith(":"):
            out.append("")
        out.append(line)
    return "\n".join(out)


def regenerate(tree: ast.AST) -> str:
    """Code generation from the AST (``ast.unparse``) with PEP 8 blank lines."""
    if not isinstance(tree, ast.Module) or not tree.body:
        return unparse(tree) + "\n"
    parts: list[str] = []
    for i, stmt in enumerate(tree.body):
        code = _blank_lines_before_nested_defs(unparse(ast.Module(body=[stmt], type_ignores=[])))
        if i:
            prev = tree.body[i - 1]
            if isinstance(stmt, _DEFS) or isinstance(prev, _DEFS):
                parts.append("\n\n\n")
            elif i == 1 and isinstance(prev, ast.Expr) and isinstance(prev.value, ast.Constant) \
                    and isinstance(prev.value.value, str):
                parts.append("\n\n")  # after the module docstring
            else:
                parts.append("\n")
        parts.append(code)
    return "".join(parts) + "\n"


def normalized_dump(source: str) -> str | None:
    try:
        return ast.dump(ast.parse(source))
    except (SyntaxError, ValueError):
        return None


# Position helpers (col offsets in the AST are UTF-8 *byte* offsets)

def _char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


@dataclass
class _Edit:
    start_line: int      # 1-based, inclusive
    start_col: int       # char column
    end_line: int        # 1-based, inclusive
    end_col: int         # char column (exclusive)
    render: Callable[[], str]
    whole_lines: bool
    method: str


def _stmt_region(lines: list[str], first: ast.stmt, last: ast.stmt) -> tuple[int, int, int, int] | None:
    """Line range covering whole statements, or None if they share lines with other code."""
    start_line = min([first.lineno] + [d.lineno for d in getattr(first, "decorator_list", [])])
    end_line = last.end_lineno
    if start_line > len(lines) or end_line > len(lines):
        return None
    head = lines[start_line - 1]
    start_col = _char_col(head, first.col_offset)
    if head[:start_col].strip():
        return None
    if isinstance(first, ast.If) and head[start_col:].startswith("elif"):
        return None  # an `elif` branch is not a standalone statement in the text
    tail =lines[end_line - 1][_char_col(lines[end_line - 1], last.end_col_offset):].strip()
    if tail and not tail.startswith("#"):
        return None
    return start_line, start_col, end_line, len(lines[end_line - 1])


def _string_continuation_lines(code: str) -> set[int]:
    """1-based line numbers that start inside a multi-line string literal."""
    inside: set[int] = set()
    open_fstrings: list[int] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            kind = tokenize.tok_name.get(tok.type, "")
            if kind == "FSTRING_START":
                open_fstrings.append(tok.start[0])
            elif kind == "FSTRING_END" and open_fstrings:
                inside.update(range(open_fstrings.pop() + 1, tok.end[0] + 1))
            elif tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                inside.update(range(tok.start[0] + 1, tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return inside


def _indent_block(code: str, indent: str) -> str:
    """Indent generated code, leaving the inside of multi-line strings untouched."""
    skip = _string_continuation_lines(code)
    return "\n".join(indent + line if line.strip() and i not in skip else line
                     for i, line in enumerate(code.split("\n"), start=1))


def _plan_edits(source: str, match: Match) -> list[_Edit]:
    """Candidate text edits, narrowest first. Must be called *before* the AST is modified."""
    lines = source.split("\n")
    edits: list[_Edit] = []
    loc = match.location

    if isinstance(loc, StmtSpan):
        nodes = loc.nodes()
        region = _stmt_region(lines, nodes[0], nodes[-1])
        if region:
            s_line, s_col, e_line, _ = region
            indent = lines[s_line - 1][:s_col]
            tail = lines[e_line - 1][_char_col(lines[e_line - 1], nodes[-1].end_col_offset):].strip()
            replacement = list(match.replacement)
            whole_block = not replacement and loc.start == 0 and loc.end == len(loc.container)

            def render(replacement=replacement, indent=indent, tail=tail, whole_block=whole_block):
                stmts = [ast.Pass()] if whole_block else replacement
                if not stmts:
                    return None  # delete the lines
                code = "\n".join(unparse(s) for s in stmts)
                code = _indent_block(code, indent)
                return code + (f"  {tail}" if tail else "")

            edits.append(_Edit(s_line, 0, e_line, len(lines[e_line - 1]), render, True, "splice"))
    else:
        node = loc.node()
        s_line, e_line = node.lineno, node.end_lineno
        s_col = _char_col(lines[s_line - 1], node.col_offset)
        e_col = _char_col(lines[e_line - 1], node.end_col_offset)
        new_expr = match.replacement
        edits.append(_Edit(s_line, s_col, e_line, e_col, lambda: unparse(new_expr), False, "splice"))
        edits.append(_Edit(s_line, s_col, e_line, e_col, lambda: f"({unparse(new_expr)})", False, "splice"))

    # Fallback: regenerate an enclosing statement that owns whole lines,
    # innermost first (every candidate is verified before it is accepted).
    for anc in match.ancestors:
        region = _stmt_region(lines, anc, anc)
        if not region:
            continue
        s_line, s_col, e_line, _ = region
        indent = lines[s_line - 1][:s_col]

        def render(anc=anc, indent=indent):
            return _indent_block(unparse(anc), indent)

        edits.append(_Edit(s_line, 0, e_line, len(lines[e_line - 1]), render, True, "statement"))
    return edits


def _apply_edit(source: str, edit: _Edit) -> str:
    lines = source.split("\n")
    new_text = edit.render()
    if edit.whole_lines:
        new_lines = [] if new_text is None else new_text.split("\n")
        k = edit.start_line - 1
        lines[k:edit.end_line] = new_lines
        if new_text is None:
            # Deleting statements must not leave a run of more than 2 blank lines
            # (or blank lines at the very top of the file).
            start = k
            while start > 0 and not lines[start - 1].strip():
                start -= 1
            end = k
            while end < len(lines) and not lines[end].strip():
                end += 1
            keep = 0 if start == 0 else min(end - start, 2)
            lines[start:end] = [""] * keep
        return "\n".join(lines)
    first, last = lines[edit.start_line - 1], lines[edit.end_line - 1]
    merged = first[:edit.start_col] + new_text + last[edit.end_col:]
    lines[edit.start_line - 1:edit.end_line] = merged.split("\n")
    return "\n".join(lines)


def apply_and_generate(source: str, tree: ast.Module, match: Match, preserve: bool = True) -> tuple[str, str]:
    """Apply ``match`` to ``tree`` (in place) and return ``(new_source, method)``."""
    edits = _plan_edits(source, match) if preserve else []
    apply_match(match)
    expected_source = regenerate(tree)
    if not preserve:
        return expected_source, "regenerate"
    expected = normalized_dump(expected_source)
    for edit in edits:
        try:
            candidate = _apply_edit(source, edit)
        except Exception:
            continue
        if normalized_dump(candidate) == expected:
            return candidate if candidate.endswith("\n") else candidate + "\n", edit.method
    return expected_source, "regenerate"


def unified_diff(before: str, after: str, name: str = "code.py") -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"original/{name}", tofile=f"refactored/{name}",
    ))
