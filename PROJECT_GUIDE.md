# How to explain this project

**One sentence:** This is a Python-to-Python refactoring tool that uses compiler techniques to
understand and rewrite code, with ML to rank proposed changes.

```text
Python source -> Parse AST -> Analyse scopes and patterns -> Rank with ML
              -> Rewrite AST -> Generate Python source -> Show results
```

Start with `pipeline.run_pipeline()`. Its four stages parse the input, detect/rank suggestions,
apply approved edits, and report results. `_rank_matches()` shares analysis and ranking;
`_refactor()` repeats rewriting because one change can enable another.

## Compiler design concepts in our code

| Concept | What our project does | Where |
|---|---|---|
| Lexical analysis | Delegates recognition of identifiers, keywords, literals and operators to Python's parser. `tokenize` also helps handle quote styles and multiline strings. | `ast_tools.parse_source`, `detect_quote_preference`; `codegen._string_continuation_lines` |
| Syntax analysis | Uses `ast.parse()` to check Python grammar, build a tree, and report syntax errors with line/column information. | `ast_tools.parse_source`, `SyntaxErrorInfo` |
| Intermediate representation | Represents the program as AST nodes such as `Assign`, `Return`, `Name` and `BinOp`; transformations work on structure. | `ast_tools.ParseResult`; the AST tab |
| Tree traversal | Uses an explicit stack for pre-order DFS, recording parents, fields and indices. | `ast_tools.dfs` |
| Semantic analysis / symbol tables | Tracks reads, writes, declarations and closure reads per scope. Rules use these checks to reject known unsafe rewrites. | `analysis.build_scopes`, `ScopeInfo`, `NodeContext` |
| Optimization | Detects 20 patterns, including constant folding, dead-code/dead-store elimination and condition simplification. | `rules.py`, R01–R20 |
| AST transformation | Replaces a matched expression or range of statements with a replacement subtree. | `analysis.ExprSlot`, `StmtSpan`; `codegen.apply_match` |
| Code generation | Turns the modified AST into Python source, optionally splicing only the changed region. | `ast_tools.unparse`; `codegen.regenerate`, `apply_and_generate` |
| Iterative optimization | Reanalyses after each edit until no approved change remains, with cycle detection and a 200-attempt limit. | `pipeline._refactor` |

This is a **source-to-source transformer**: Python goes in and Python comes out. We reuse Python's
parser; we have not built a separate lexer, grammar, machine-code generator, full control-flow graph
or full type checker. The AST is our intermediate representation; there is no three-address-code stage.

## Explain one example

Input:

```python
def add(a, b):
    result = a + b
    return result
```

1. Parsing creates a `FunctionDef` containing an `Assign` followed by a `Return`.
2. DFS reaches the statement block. **R01** recognizes that the assigned variable is immediately returned.
3. Scope checks ensure that the variable is function-local, without a declaration or detected
   closure capture that would prevent this rewrite.
4. Feature extraction measures the match, and ML scores whether it is worth applying.
5. If approved, the two statements become one `Return` containing the original `BinOp(a + b)`.
6. Code generation emits the new source, reparses it, and compares its AST with the intended result.

Output:

```python
def add(a, b):
    return a + b
```

Two direct compiler optimization examples are **R08**, which folds `3 * 4` into `12`, and **R09**,
which removes statements after an unconditional `return`. A loop-to-comprehension rewrite (**R04**)
can enable R01 in the next iteration.

## What the AI contributes

Rules detect opportunities and construct replacements. ML ranks them; it does not generate arbitrary
code or prove correctness. A classifier takes **19 features**, including rule ID, risk, node/line
reduction, resulting line length and nesting. By default, scores of at least **0.5** select changes.
Users can override decisions or apply all suggestions.

Training compares Random Forest, Gradient Boosting and Logistic Regression using five-fold
cross-validation. Twenty template families generate examples; a reviewer policy supplies labels
with 3% simulated disagreement. Optional manual labels are supported. A valid but deeply nested
comprehension or long conditional expression may receive a low readability score.

Semantic checks are partial: dynamic types and side effects prevent a general equivalence guarantee.
Reparsing verifies syntax/tree structure. Separate tests compare original/refactored behavior on
examples. ML scores estimate agreement with reviewer-policy labels, not proof of equivalence.

## The seven engine modules

| Module | Role to remember |
|---|---|
| `pipeline.py` | Connect the stages |
| `ast_tools.py` | Parse, traverse, measure, draw and render ASTs |
| `analysis.py` | Scope information, matches and ML features |
| `rules.py` | Twenty refactoring patterns and their checks |
| `codegen.py` | Apply replacements and verify generated source |
| `ml_model.py` | Train, compare, save, load and score classifiers |
| `dataset.py` | Generate and label training examples |

`app.py` handles the interface. `samples/`, `tests/`, `data/`, `models/` and the training script remain.
Parsing/rendering helpers formerly in `parser.py` and `unparsing.py` now live in `ast_tools.py`;
features formerly in `features.py` now live in `analysis.py`. The public API remains
`from refactor_engine import run_pipeline`.

All 20 rules, UI controls, AST views, formatting modes, training options and example programs are
retained. Rules share assignment/location helpers, the pipeline shares ranking/decision logic,
and the trained model uses compression without changing predictions. The remaining guards and
source-edit fallbacks support those features.

## A short presentation script

“Our project accepts Python code and parses it into an Abstract Syntax Tree. We traverse the tree
using DFS and build symbol information to track variable usage. Twenty rules detect refactoring
patterns, including constant folding, dead-code removal and simpler conditions. A machine-learning
classifier ranks the suggestions using features extracted from the AST. We apply an approved
replacement, generate Python source and verify its structure by parsing it again. We repeat the
process because one improvement can enable another. The interface shows the original and refactored
code, explains suggestions and visualizes the AST. This demonstrates a compiler pipeline that
translates Python into cleaner Python.”
