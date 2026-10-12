"""End-to-end tests of the pipeline, code generation, ML ranker and AST tools."""

import ast
from pathlib import Path

import pytest

from refactor_engine import run_pipeline
from refactor_engine.ast_tools import dfs, to_graphviz
from refactor_engine.pipeline import get_ranker, traversal_table
from behaviour import compare_behaviour

SAMPLES = sorted((Path(__file__).resolve().parent.parent / "samples").glob("*.py"))


def test_report_example():
    """The exact example from the project understanding report (section 5)."""
    code = "def add(a, b):\n    result = a + b\n    return result\n"
    result = run_pipeline(code)
    assert result.refactored_source == "def add(a, b):\n    return a + b\n"
    (s,) = result.suggestions
    assert s.rule_id == "R01"
    assert "unnecessary because it is used only for returning the value" in s.title


@pytest.mark.parametrize("path", SAMPLES, ids=[p.stem for p in SAMPLES])
@pytest.mark.parametrize("mode", ["recommended", "all"])
def test_samples_preserve_behaviour(path, mode):
    source = path.read_text(encoding="utf-8")
    result = run_pipeline(source, apply_mode=mode)
    assert result.ok
    ast.parse(result.refactored_source)
    outcome = compare_behaviour(result.source, result.refactored_source)
    assert outcome["same"], (outcome["original"].stdout, outcome["refactored"].stdout)


@pytest.mark.parametrize("path", SAMPLES, ids=[p.stem for p in SAMPLES])
def test_refactoring_reaches_fixed_point(path):
    once = run_pipeline(path.read_text(encoding="utf-8")).refactored_source
    assert run_pipeline(once).refactored_source == once


def test_comments_are_preserved():
    code = (
        "# header comment\n"
        "def add(a, b):\n"
        "    # explain\n"
        "    result = a + b  # sum\n"
        "    return result\n"
        "\n"
        "\n"
        "x = 2 + 3  # five\n"
    )
    out = run_pipeline(code).refactored_source
    assert "# header comment" in out and "# explain" in out and "# five" in out
    assert "return a + b" in out and "x = 5" in out


def test_regenerate_mode_drops_comments_with_warning():
    result = run_pipeline("x = 2 + 3  # five\n", preserve_formatting=False)
    assert result.refactored_source == "x = 5\n"
    assert any("comments" in w for w in result.warnings)


def test_syntax_error_reported():
    result = run_pipeline("def broken(:\n    pass\n")
    assert not result.ok and result.parse.error.line == 1
    assert result.suggestions == []


def test_indented_paste_is_accepted():
    result = run_pipeline("    def f(x):\n        y = x\n        return y\n")
    assert result.ok and "return x" in result.refactored_source


def test_analysis_only_mode_changes_nothing():
    code = "def add(a, b):\n    result = a + b\n    return result\n"
    result = run_pipeline(code, apply_mode="none")
    assert result.suggestions and not result.steps and result.refactored_source == code


def test_override_disables_a_suggestion():
    code = "def add(a, b):\n    result = a + b\n    return result\n"
    first = run_pipeline(code)
    key = first.suggestions[0].key
    result = run_pipeline(code, overrides={key: False})
    assert result.refactored_source == code


def test_follow_up_refactorings_are_applied():
    code = ("def f(xs):\n"
            "    out = []\n"
            "    for x in xs:\n"
            "        out.append(x * 2)\n"
            "    return out\n")
    result = run_pipeline(code)
    assert result.refactored_source == "def f(xs):\n    return [x * 2 for x in xs]\n"
    assert [s.rule_id for s in result.steps] == ["R04", "R01"]


def test_original_ast_remains_available_for_the_viewer():
    code = "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x * 2)\n    return out\n"
    result = run_pipeline(code)
    assert ast.dump(result.parse.tree) == ast.dump(ast.parse(code))
    assert not any(isinstance(node, ast.For) for node in ast.walk(result.refactored_tree))
    original_ids = {id(node) for node in ast.walk(result.parse.tree)}
    assert result.suggestions[0].highlight_ids <= original_ids


def test_override_can_enable_a_change_below_the_threshold():
    code = "def add(a, b):\n    result = a + b\n    return result\n"
    first = run_pipeline(code, threshold=1.1)
    assert not first.changed
    result = run_pipeline(code, threshold=1.1, overrides={first.suggestions[0].key: True})
    assert result.refactored_source == "def add(a, b):\n    return a + b\n"


def test_ml_ranker_scores_and_prefers_readable_code():
    ranker = get_ranker()
    assert ranker.available
    simple = run_pipeline("def f(a, b):\n    r = a + b\n    return r\n").suggestions[0]
    complex_ = run_pipeline(
        "def f(principal, rate, years):\n"
        "    monthly_installment = principal * (rate / 12) / (1 - (1 + rate / 12) ** (-years * 12))\n"
        "    return monthly_installment\n").suggestions[0]
    assert simple.score > 0.5 > complex_.score


def test_dfs_is_preorder_left_to_right():
    tree = ast.parse("a = 1\nb = 2\n")
    order = [type(e.node).__name__ for e in dfs(tree, include_trivial=False)]
    assert order == ["Module", "Assign", "Name", "Constant", "Assign", "Name", "Constant"]
    assert len(traversal_table(tree, include_trivial=True)) == len(dfs(tree))


def test_graphviz_output():
    tree = ast.parse("def f(x):\n    return x + 1\n")
    dot, count, truncated = to_graphviz(tree, highlight={id(tree.body[0])})
    assert dot.startswith("digraph AST") and "FunctionDef: f" in dot and "#dc2626" in dot
    assert count > 5 and not truncated
    _, count, truncated = to_graphviz(tree, max_nodes=3)
    assert count == 3 and truncated


def test_unicode_source_positions():
    code = 'def f():\n    s = "héllo wörld" + "!"\n    return s\n'
    out = run_pipeline(code, apply_mode="all").refactored_source
    assert out == 'def f():\n    return "héllo wörld!"\n'


def test_elif_branch_and_docstring_regeneration():
    code = (
        "def f(x):\n"
        '    """Doc\n'
        '    string."""\n'
        "    # important\n"
        "    if x == 1:\n"
        '        return "a"\n'
        "    elif x == 2:\n"
        '        return "b"\n'
        "    else:\n"
        "        print(x)\n"
        "        return x\n"
        "\n"
        "\n"
        "# tail comment\n"
        "print(f(1), f(2), f(3))\n"
    )
    result = run_pipeline(code, apply_mode="all")
    out = result.refactored_source
    assert "# important" in out and "# tail comment" in out
    assert all(s.method != "regenerate" for s in result.steps)
    assert ast.get_docstring(ast.parse(out).body[0]) == "Doc\nstring."
    assert compare_behaviour(code, out)["same"]
