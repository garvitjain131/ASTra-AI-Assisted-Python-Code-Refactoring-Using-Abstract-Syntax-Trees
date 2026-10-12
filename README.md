# AI-Based Code Refactoring Using Compiler AST

Compiler Design mini project by Garvit Jain (E225), Aaryan Jain (E249) and Pushp Jain (E276).

For a short walkthrough and presentation script, read [PROJECT_GUIDE.md](PROJECT_GUIDE.md).
The engine now has seven modules instead of ten, with all 20 refactorings retained.

You paste or upload Python code. The tool builds the code's **Abstract Syntax Tree**, walks it with
**DFS**, and **pattern-matches** 20 refactoring opportunities. A **machine-learning model** ranks
them, and the tool then **transforms the AST** and **generates** cleaner code that you can copy or
download. The tool also explains every suggestion and visualises the AST.

```
Input → Python Parser → AST Analysis (DFS + pattern matching) → AI/ML ranking
      → Refactoring (AST transformation) → Code Generation (ast.unparse) → Output
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py                         # web UI at http://localhost:8501
```

The trained model is in `models/`. If it is missing, it is trained automatically on first use (about 30 s).

```bash
python scripts/train_model.py     # rebuild the dataset and retrain the model
python -m pytest                  # run the tests
```

## Using the app

1. **Add code:** paste it into the editor, pick an example from **Load an example**, or use **Upload .py file**.
   Then press **Refactor code**. Results appear only after this click, and are hidden again as soon
   as the code in the editor changes, so what you see always belongs to the code in the editor.
2. **Read the results** in three tabs:
   * **Refactored code**: original and refactored code side by side. Use the copy icon in the
     top-right corner of the code box, or **Download refactored code**.
   * **Suggestions**: every detected issue with an explanation, before/after code and the ML
     model's confidence. Use **Apply** to include or exclude a change, and **Reset choices** to undo
     your switches.
   * **AST**: the Abstract Syntax Tree as a graph (original or refactored, whole program or one
     function), with nodes involved in a suggestion in red. It also shows the tree as text and the
     DFS traversal order.
3. **Settings** (optional): apply only ML-recommended changes (default) or all suggestions, and keep or
   drop comments and formatting.

**Clear** empties the editor and the results.

## Example (from the project report)

```python
def add(a, b):
    result = a + b
    return result
```

**Suggestion:** The variable `result` is unnecessary because it is used only for returning the value.

```python
def add(a, b):
    return a + b
```

## Supported refactorings

| ID | Refactoring | Example |
|---|---|---|
| R01 | Redundant temporary variable | `r = a + b; return r` → `return a + b` |
| R02 | Boolean return simplification | `if c: return True else: return False` → `return c` |
| R03 | Comparison with None/True/False | `x == None` → `x is None`, `if ok == True` → `if ok` |
| R04 | Loop → list/set/dict comprehension | `out = []; for x in xs: if p: out.append(f)` → `out = [f for x in xs if p]` |
| R05 | Accumulator loop → `sum()` | `t = 0; for x in xs: t += x` → `t = sum(xs)` |
| R06 | `range(len())` loop | `for i in range(len(xs)): print(xs[i])` → `for x in xs: print(x)` / `enumerate` |
| R07 | Augmented assignment | `n = n + 1` → `n += 1` |
| R08 | Constant folding *(code optimisation)* | `x = 3 * 4` → `x = 12` |
| R09 | Unreachable code *(dead-code elimination)* | statements after `return`/`raise`/`break`/`continue` |
| R10 | Merge nested `if` | `if a: if b:` → `if a and b:` |
| R11 | Conditional expression | `if c: x = a else: x = b` → `x = a if c else b` |
| R12 | Redundant `else` after `return` | guard-clause style |
| R13 | Unused import | `import os` (never used) removed |
| R14 | `dict.get()` with default | `if k in d: v = d[k] else: v = 0` → `v = d.get(k, 0)` |
| R15 | Negated comparison | `not a in b` → `a not in b` |
| R16 | Unused local variable *(dead store)* | side-effect-free assignment never read |
| R17 | Empty collection check | `len(xs) == 0` → `not xs` |
| R18 | Redundant `pass` | `pass` in a non-empty block |
| R19 | Membership test | `x == 1 or x == 2` → `x in (1, 2)` |
| R20 | Chained comparison | `0 <= a and a < 10` → `0 <= a < 10` |

## How it works

### 1. Parser & AST generator (`ast_tools.py`)
The input is normalised: BOM removed, newlines unified, tabs expanded and common indentation removed,
so a snippet pasted from inside a class still parses. Python's own parser (`ast.parse`) then does the
lexical and syntax analysis. Syntax errors are reported with their line and column.

### 2. AST analysis (`ast_tools.py`, `analysis.py`)
* **Iterative pre-order DFS** with an explicit stack (no recursion limit), visiting children left to
  right. Each node is recorded with its parent, field and index (nodes and edges).
* **Symbol tables per scope** (module, function, class, lambda) record which names are read and
  written, which are declared `global`/`nonlocal`, and which are captured by closures.
* Every node and every statement block is handed to the **rules** (`rules.py`). These pattern-match
  on node shapes and use the symbol tables to prove the rewrite is safe. For example, R01 is
  rejected if a closure reads the variable, R04 inside a class body (comprehensions cannot see class
  variables), R06 if the loop writes `seq[i]`, and R05 if `sum` is shadowed. Each match gets a
  **risk** score.

### 3. AI/ML ranking (`analysis.py`, `dataset.py`, `ml_model.py`)
Some valid refactorings still make code worse. Examples are a 3-level nested comprehension, a
100-character conditional expression, replacing a well-named explaining variable, or folding
`60 * 60 * 24` into `86400`. The ML model decides which suggestions are worth applying.

* **Feature extraction:** each candidate becomes 19 features. These are the rule id, risk, AST
  nodes/lines before and after, resulting line length, replacement depth, nesting and loop depth,
  calls, boolean operators, comprehension shape, variable-name length, and whether the code is
  inside a function.
* **Dataset:** the dataset is built by *weak supervision*. 20 template generators write thousands of
  varied programs, the analyser extracts the candidates, and an *expert policy* labels them. The
  policy covers PEP 8 line length, no deep comprehensions, keeping explanatory variables, never
  auto-applying risky rewrites, and so on. 3% of labels are flipped to simulate reviewer
  disagreement. The current set has about 3,900 rows (`data/training_data.csv`). Hand-labelled rows
  can be added to `data/manual_labels.csv`.
* **Models:** Random Forest, Gradient Boosting and Logistic Regression (one-hot rule id + scaled
  numeric features) are compared with 5-fold cross-validation, and the best by F1 is kept. Current
  hold-out results are about 0.97 accuracy and 0.98 F1 (`models/metrics.json`).
* The predicted probability is the **ML score**. It ranks the suggestions, and those at or above the
  threshold (0.5 by default) are applied automatically. Sample `04_readability_tradeoffs.py` shows
  valid refactorings that the model declines.

### 4. Refactoring (`codegen.py`, `pipeline.py`)
Rule-based **AST transformation**: the matched statements or expression are replaced by the
replacement subtree. Refactoring runs to a **fixed point**. The highest-scoring approved match is
applied, the code is regenerated and re-parsed, and analysis runs again, so one change can enable
another (loop → comprehension → redundant temporary removed).

### 5. Code generation (`codegen.py`, `ast_tools.py`)
* **Preserve mode (default):** only the changed region is generated with `ast.unparse` and spliced
  into the original text using the AST position information. Comments and formatting elsewhere are
  kept.
* **Regenerate mode:** the whole program is produced from the AST with `ast.unparse`, using PEP 8
  blank lines.
* **Verification:** every edit is re-parsed, and its AST must be identical to the transformed AST.
  If it is not, a wider enclosing statement is regenerated instead, so a wrong splice can never
  reach the output.

### Behaviour preservation
* Scope analysis rejects known unsafe patterns, but does not prove general equivalence. Rewrites that rely on
  assumptions, such as `x == True` → `x` or `len(x) == 0` → `not x`, carry a higher risk score and
  are labelled *Review*.
* The test suite runs every sample before and after refactoring and checks the output is identical.

## Project structure

```
app.py                     Streamlit UI
refactor_engine/
  ast_tools.py             Parser, DFS, metrics, Graphviz view and source rendering
  analysis.py              Scope/symbol tables, matches and ML feature extraction
  rules.py                 20 pattern-matching refactoring rules
  dataset.py               Synthetic dataset generator + expert labelling policy
  ml_model.py              Model training, comparison, persistence, ranking
  codegen.py               AST transformation, splicing code generation, verification
  pipeline.py              End-to-end orchestration (fixed-point refactoring)
models/                    Trained model + metrics
data/                      Generated training data
samples/                   Example programs
scripts/train_model.py     Training script
tests/                     pytest tests
```

## Limitations
* Only Python and only the 20 rules above are supported. This is not a general optimiser.
* Type information is not available, so rules that depend on types (`+=` on lists, truthiness of
  NumPy arrays) are flagged with a higher risk instead of being proven safe.
* In regenerate mode comments are lost, because they are not part of the AST.
