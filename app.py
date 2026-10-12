"""Streamlit interface - AI-Based Code Refactoring Using Compiler AST.

Run with:   streamlit run app.py
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from refactor_engine.ast_tools import count_nodes, scope_nodes, to_graphviz
from refactor_engine.pipeline import get_ranker, run_pipeline, traversal_table

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
MAX_GRAPH_NODES = 300

st.set_page_config(page_title="AST Code Refactoring", page_icon="🌳", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(
    """<style>
    /* Streamlit leaves ~6rem above the first element; start the page just below the toolbar. */
    [data-testid="stMainBlockContainer"], .block-container {padding-top: 2.5rem !important;}
    [data-testid="stHeader"] {height: 2.5rem; background: transparent;}
    h1 {padding-top: 0 !important;}
    @media (max-width: 640px) {h1 {font-size: 1.9rem !important;}}
    textarea {font-family: 'Source Code Pro', ui-monospace, Consolas, monospace !important;
              font-size: 14px !important;}
    </style>""",
    unsafe_allow_html=True,
)


# State

def load_samples() -> dict[str, str]:
    samples = {}
    for path in sorted(SAMPLES_DIR.glob("*.py")):
        title = path.stem.split("_", 1)[-1].replace("_", " ").capitalize()
        samples[title] = path.read_text(encoding="utf-8")
    return samples


SAMPLES = load_samples()

ss = st.session_state
ss.setdefault("code_input", "")
ss.setdefault("submitted", None)  # code of the last "Refactor code" click; results are shown only for it
ss.setdefault("overrides", {})
ss.setdefault("upload_nonce", 0)


@st.cache_resource(show_spinner=False)
def ranker():
    return get_ranker()


@st.cache_resource(show_spinner=False, max_entries=32)
def analyse(source: str, apply_mode: str, preserve: bool, overrides: tuple):
    return run_pipeline(source, apply_mode=apply_mode, preserve_formatting=preserve,
                        overrides=dict(overrides), ranker=ranker())


# Button / widget callbacks (run before the page is redrawn)

def submit():
    ss.submitted = ss.code_input
    ss.overrides = {}
    ss.pop("ast_scope", None)  # new code: start the AST view on its default scope


def on_example():
    name = ss.get("example")
    if name:
        ss.code_input = SAMPLES[name]  # only fills the editor; results come after "Refactor code"
        ss.example = None  # reset, so the same example can be picked again


def on_upload():
    file = ss.get(f"upload_{ss.upload_nonce}")
    if file is not None:
        ss.code_input = file.getvalue().decode("utf-8", errors="replace")
        ss.upload_nonce += 1  # fresh, empty uploader: the same file can be uploaded again


def on_clear():
    ss.code_input = ""
    ss.submitted = None
    ss.overrides = {}


def on_toggle(key: tuple, widget_key: str):
    ss.overrides = {**ss.overrides, key: ss[widget_key]}


def on_reset_choices():
    ss.overrides = {}


def toggle_key(key: tuple, occurrence: int) -> str:
    digest = hashlib.md5(repr(key).encode("utf-8")).hexdigest()[:12]
    return f"apply_{digest}_{occurrence}"


# Header + input

st.title("AI-Based Code Refactoring Using Compiler AST")
st.caption("Paste Python code: it is parsed into an Abstract Syntax Tree, analysed for refactoring patterns, "
           "ranked by a machine-learning model and rewritten into cleaner code.")

with st.container(border=True):
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    c1.selectbox("Load an example", list(SAMPLES), index=None, placeholder="Choose an example program...",
                 key="example", on_change=on_example)
    with c2.popover("Upload .py file", width="stretch"):
        st.file_uploader("Python file", type=["py", "txt"], key=f"upload_{ss.upload_nonce}",
                         on_change=on_upload, label_visibility="collapsed")
    with c3.popover("Settings", icon=":material/tune:", width="stretch"):
        st.radio("Apply", ["Recommended by ML", "All suggestions"], key="apply_choice",
                 help="Recommended: only changes the ML model scores at 0.5 or higher.")
        st.toggle("Keep comments and formatting", value=True, key="preserve",
                  help="Off: the whole program is regenerated from the AST with ast.unparse, which drops comments.")

    st.text_area("Your Python code", key="code_input", height=280, label_visibility="collapsed",
                 placeholder="Paste your Python code here, e.g.\n\ndef add(a, b):\n    result = a + b\n    return result")
    b1, b2, _ = st.columns([1, 1, 4], vertical_alignment="center")
    b1.button("Refactor code", type="primary", icon=":material/auto_fix_high:", on_click=submit,
              width="stretch", key="run_btn")
    b2.button("Clear", icon=":material/delete:", on_click=on_clear, width="stretch", key="clear_btn")

# Results are shown only after "Refactor code" is pressed, and only for the code it was pressed on.
if not ss.code_input.strip():
    st.info("Paste Python code above (or load an example) and press **Refactor code**.")
    st.stop()
if ss.submitted != ss.code_input:
    st.info("Press **Refactor code** to analyse this code.")
    st.stop()
source = ss.submitted

apply_mode = "all" if ss.get("apply_choice") == "All suggestions" else "recommended"
with st.spinner("Parsing, analysing and refactoring..."):
    result = analyse(source, apply_mode, ss.get("preserve", True), tuple(sorted(ss.overrides.items())))

if not result.ok:
    err = result.parse.error
    st.error(f"This code has a syntax error, so no AST could be built: **{err.message}** (line {err.line or '?'})")
    if err.line:
        lines = result.source.splitlines()
        lo, hi = max(0, err.line - 3), min(len(lines), err.line + 2)
        st.code("\n".join(f"{'>' if i + 1 == err.line else ' '} {i + 1:3d} | {lines[i]}" for i in range(lo, hi)),
                language="text")
    st.stop()

# Summary

mb, ma = result.metrics_before, result.metrics_after


def delta(key: str) -> str | None:
    d = ma[key] - mb[key]
    return None if d == 0 else f"{d:+d}"


m1, m2, m3, m4 = st.columns(4)
m1.metric("Issues found", len(result.suggestions))
m2.metric("Changes applied", len(result.steps),
          help="Can be more than the issues found: applying one change can make a follow-up change possible.")
m3.metric("Lines of code", ma["code_lines"], delta("code_lines"), delta_color="inverse")
m4.metric("Cyclomatic complexity", ma["complexity"], delta("complexity"), delta_color="inverse")

for w in result.warnings:
    st.warning(w)

tab_code, tab_sugg, tab_ast = st.tabs(["Refactored code", f"Suggestions ({len(result.suggestions)})", "AST"])

# Tab 1: refactored code

with tab_code:
    if not result.suggestions:
        st.success("No refactoring opportunities found - the code already looks clean.")
    elif not result.changed:
        st.info("Suggestions were found, but none are applied. Switch some on in the **Suggestions** tab, "
                "or choose **All suggestions** in Settings.")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Original**")
        st.code(result.source, language="python", line_numbers=True)
    with right:
        st.markdown("**Refactored**")
        st.code(result.refactored_source, language="python", line_numbers=True)
    st.caption("To copy, hover over the code and click the copy icon in its top-right corner.")
    st.download_button("Download refactored code", result.refactored_source, file_name="refactored.py",
                       mime="text/x-python", icon=":material/download:", on_click="ignore", key="download_btn")

# Tab 2: suggestions

with tab_sugg:
    if not result.suggestions:
        st.info("No refactoring opportunities detected.")
    else:
        h1, h2 = st.columns([4, 1], vertical_alignment="center")
        h1.caption("Each suggestion shows the detected issue and the ML model's confidence that the change "
                   "improves the code. Use **Apply** to include or exclude a change.")
        h2.button("Reset choices", on_click=on_reset_choices, disabled=not ss.overrides, width="stretch",
                  key="reset_btn")
        seen: dict[tuple, int] = {}
        for s in result.suggestions:
            occurrence = seen.get(s.key, 0)
            seen[s.key] = occurrence + 1
            with st.container(border=True):
                top, switch = st.columns([5, 1], vertical_alignment="center")
                where = f"line {s.line}" if s.line == s.end_line else f"lines {s.line}-{s.end_line}"
                top.markdown(f"**{s.rule_name}** · {where}")
                wkey = toggle_key(s.key, occurrence)
                ss[wkey] = s.will_apply
                switch.toggle("Apply", key=wkey, on_change=on_toggle, args=(s.key, wkey))
                st.write(s.title)
                label = "recommended" if s.recommended else "not recommended"
                st.progress(min(max(s.score, 0.0), 1.0), text=f"ML confidence {s.score:.2f} ({label})")
                before, after = st.columns(2)
                before.caption("Before")
                before.code(s.before_code, language="python")
                after.caption("After")
                after.code(s.after_code, language="python")
                st.caption(s.explanation)

        info = result.model_info
        if info.get("available") and "accuracy" in info:
            with st.expander("How suggestions are ranked"):
                st.markdown(
                    f"Each detected pattern is turned into features taken from the AST (pattern type, risk, "
                    f"nodes and lines before/after, resulting line length, nesting depth...). A "
                    f"**{info['model']}** classifier predicts how likely the change is to improve the code. "
                    f"Changes scoring 0.5 or higher are applied automatically.\n\n"
                    f"Test accuracy **{info['accuracy']:.1%}**, F1 score **{info['f1']:.3f}** "
                    f"({info.get('training_rows', 0):,} training examples)."
                )

# Tab 3: AST

with tab_ast:
    a1, a2 = st.columns([1, 2], vertical_alignment="bottom")
    which = a1.radio("Tree", ["Original", "Refactored"], horizontal=True, key="ast_tree")
    tree = result.parse.tree if which == "Original" else result.refactored_tree
    choices = scope_nodes(tree)
    labels = [label for label, _ in choices]
    # Keep the chosen scope valid when the code or tree changes. A whole program
    # is too wide to read as one graph, so larger programs start on their first function.
    if ss.get("ast_scope") not in labels:
        too_big = count_nodes(tree) > 80
        ss.ast_scope = labels[1] if too_big and len(labels) > 1 else labels[0]
    a2.selectbox("Show", labels, key="ast_scope")
    root = dict(choices)[ss.ast_scope]

    highlight = set()
    if which == "Original":
        for s in result.suggestions:
            highlight |= s.highlight_ids
    dot, shown, truncated = to_graphviz(root, highlight, max_nodes=MAX_GRAPH_NODES)
    st.caption("Blue: statements · purple: definitions · green: expressions · amber: literals"
               + (" · **red: code involved in a suggestion**" if highlight else ""))
    if truncated:
        st.warning(f"Large tree: showing the first {shown} nodes. Choose a single function above to see it clearly.")
    st.graphviz_chart(dot, width="stretch")

    with st.expander("AST as text (ast.dump)"):
        st.code(ast.dump(root, indent=2), language="python")
    with st.expander("DFS traversal order"):
        rows = traversal_table(root)
        st.caption(f"Pre-order depth-first search visits {len(rows)} nodes, children left to right.")
        df = pd.DataFrame(rows)[["order", "depth", "node", "field", "line"]]
        df["node"] = ["    " * d + n for d, n in zip(df["depth"], df["node"])]
        df["line"] = df["line"].astype("Int64")  # blank instead of "None" for nodes without a line
        st.dataframe(df, hide_index=True, width="stretch", height=380)
