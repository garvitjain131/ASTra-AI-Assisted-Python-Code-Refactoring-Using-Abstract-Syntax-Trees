"""Generate varied Python examples and label AST matches using a reviewer policy."""

from __future__ import annotations

import random

import pandas as pd

from .analysis import Analyzer, FEATURE_COLUMNS, extract_features
from .ast_tools import parse_source
from .rules import ALL_RULES

SHORT = ["x", "y", "n", "v", "r", "t", "tmp", "res", "val", "out"]
DESCRIPTIVE = ["total_price", "average_speed", "normalized_score", "monthly_interest", "user_count",
               "discount_rate", "elapsed_seconds", "weighted_mean", "final_balance", "net_salary",
               "result", "answer", "value", "total"]
ARGS = ["a", "b", "c", "price", "qty", "rate", "amount", "count", "width", "height", "base", "level"]
SEQS = ["items", "values", "numbers", "scores", "prices", "users", "records", "words", "data", "rows"]
FUNCS = ["abs", "round", "len", "str", "int", "float", "max", "min"]
MODULES = ["os", "sys", "math", "json", "re", "random", "time", "collections", "itertools", "string"]


def rexpr(rng: random.Random, names: list[str], depth: int) -> str:
    """Random arithmetic/call expression over ``names``."""
    if depth <= 0 or rng.random() < 0.25:
        return rng.choice(names + [str(rng.randint(1, 100))])
    kind = rng.random()
    if kind < 0.55:
        op = rng.choice(["+", "-", "*", "/", "//", "%"])
        return f"{rexpr(rng, names, depth - 1)} {op} {rexpr(rng, names, depth - 1)}"
    if kind < 0.75:
        f = rng.choice(["abs", "round", "float", "int"])
        return f"{f}({rexpr(rng, names, depth - 1)})"
    if kind < 0.9:
        f = rng.choice(["max", "min"])
        return f"{f}({rexpr(rng, names, depth - 1)}, {rexpr(rng, names, depth - 1)})"
    return f"({rexpr(rng, names, depth - 1)})"


def rcond(rng: random.Random, names: list[str], parts: int) -> str:
    conds = []
    for _ in range(parts):
        a = rng.choice(names)
        op = rng.choice([">", "<", ">=", "<=", "==", "!="])
        b = rng.choice(names + [str(rng.randint(0, 50))])
        conds.append(f"{a} {op} {b}")
    return f" {rng.choice(['and', 'or'])} ".join(conds)


def pick_var(rng: random.Random) -> str:
    return rng.choice(SHORT if rng.random() < 0.5 else DESCRIPTIVE)


def indent(code: str, level: int = 1) -> str:
    pad = "    " * level
    return "\n".join(pad + l if l else l for l in code.split("\n"))


# --- one generator per family of patterns ----------------------------------

def t_temp_return(rng):
    args = rng.sample(ARGS, rng.randint(1, 4))
    var = pick_var(rng)
    pre = f"    print({args[0]})\n" if rng.random() < 0.3 else ""
    return f"def f({', '.join(args)}):\n{pre}    {var} = {rexpr(rng, args, rng.randint(1, 5))}\n    return {var}\n"


def t_bool_return(rng):
    args = rng.sample(ARGS, 3)
    cond = rcond(rng, args, rng.randint(1, 5)) if rng.random() < 0.7 else rng.choice(args)
    a, b = ("True", "False") if rng.random() < 0.6 else ("False", "True")
    if rng.random() < 0.5:
        body = f"    if {cond}:\n        return {a}\n    else:\n        return {b}\n"
    else:
        body = f"    if {cond}:\n        return {a}\n    return {b}\n"
    return f"def check({', '.join(args)}):\n{body}"


def t_singleton(rng):
    v = rng.choice(ARGS + SEQS)
    kind = rng.choice(["None", "None", "True", "False"])
    op = rng.choice(["==", "!="])
    if rng.random() < 0.3:
        cond = f"{v} {op} {kind} and {rcond(rng, ARGS, rng.randint(1, 3))}"
    else:
        cond = f"{v} {op} {kind}"
    return f"def g({v}, a, b, c):\n    if {cond}:\n        return 1\n    return 0\n"


def t_comprehension(rng):
    seq = rng.choice(SEQS)
    item = rng.choice(["x", "item", "value", "entry", "n"])
    var = rng.choice(["result", "out", "filtered", "squares", "names", "selected"])
    kind = rng.choice(["list", "list", "list", "set", "dict"])
    nest = rng.choices([1, 2, 3], weights=[6, 3, 1])[0]
    has_if = rng.random() < 0.5
    expr = rexpr(rng, [item], rng.randint(0, 6))
    init = {"list": "[]", "set": "set()", "dict": "{}"}[kind]
    leaf = {"list": f"{var}.append({expr})", "set": f"{var}.add({expr})", "dict": f"{var}[{item}] = {expr}"}[kind]
    lines, level = [f"{var} = {init}"], 0
    loop_vars = [item, "j", "k"]
    for d in range(nest):
        src = seq if d == 0 else f"range({loop_vars[d - 1]})"
        lines.append("    " * level + f"for {loop_vars[d]} in {src}:")
        level += 1
    if has_if:
        lines.append("    " * level + f"if {rcond(rng, [item], rng.randint(1, 3))}:")
        level += 1
    lines.append("    " * level + leaf)
    after = f"print({item})\n" if rng.random() < 0.15 else ""
    lines.append(after + f"return {var}")
    return f"def build({seq}):\n" + indent("\n".join(lines)) + "\n"


def t_sum(rng):
    seq = rng.choice(SEQS)
    item = rng.choice(["x", "item", "value", "n", "p"])
    var = rng.choice(["total", "count", "acc", "s", "running_sum"])
    init = rng.choice(["0", "0", "0.0"])
    expr = item if rng.random() < 0.4 else rexpr(rng, [item], rng.randint(1, 5))
    if rng.random() < 0.4:
        loop_body = f"        if {rcond(rng, [item], rng.randint(1, 3))}:\n            {var} += {expr}\n"
    else:
        loop_body = f"        {var} += {expr}\n"
    after = f"    print({item})\n" if rng.random() < 0.15 else ""
    return f"def agg({seq}):\n    {var} = {init}\n    for {item} in {seq}:\n{loop_body}{after}    return {var}\n"


def t_range_len(rng):
    seq = rng.choice(SEQS)
    use_index = rng.random() < 0.4
    body = f"        print({seq}[i])\n"
    if use_index:
        body += "        print(i)\n"
    if rng.random() < 0.3:
        body += f"        total = {seq}[i] * 2\n"
    return f"def show({seq}):\n    for i in range(len({seq})):\n{body}"


def t_augassign(rng):
    v = rng.choice(ARGS + SHORT)
    op = rng.choice(["+", "-", "*", "/", "//", "%", "|"])
    rhs = str(rng.randint(1, 9)) if rng.random() < 0.5 else rexpr(rng, ARGS, rng.randint(0, 3))
    return f"def step({v}, a, b, c, price, qty):\n    {v} = {v} {op} {rhs}\n    return {v}\n"


def t_constfold(rng):
    choice = rng.random()
    if choice < 0.3:
        expr = f"{rng.randint(1, 99)} {rng.choice(['+', '-', '*'])} {rng.randint(1, 99)}"
    elif choice < 0.5:
        expr = " * ".join(str(rng.choice([60, 60, 24, 7, 1000, 365, 12])) for _ in range(rng.randint(2, 4)))
    elif choice < 0.7:
        expr = f"{rng.randint(1, 50)} / {rng.randint(1, 13)}"
    elif choice < 0.85:
        expr = f"'{rng.choice(['ab', 'Hello, ', 'x'])}' + '{rng.choice(['cd', 'World', 'y'])}'"
    else:
        expr = f"{rng.randint(2, 10)} ** {rng.randint(2, 8)}"
    var = rng.choice(["LIMIT", "SECONDS", "ratio", "label", "size", "TIMEOUT"])
    return f"{var} = {expr}\nprint({var})\n"


def t_dead_code(rng):
    args = rng.sample(ARGS, 2)
    extra = "\n".join(f"    print({rng.choice(args)})" for _ in range(rng.randint(1, 4)))
    kw = rng.choice(["return " + rexpr(rng, args, 1), "raise ValueError('bad')"])
    return f"def h({', '.join(args)}):\n    {kw}\n{extra}\n"


def t_nested_if(rng):
    args = rng.sample(ARGS, 4)
    c1 = rcond(rng, args, rng.randint(1, 3))
    c2 = rcond(rng, args, rng.randint(1, 4))
    return f"def n({', '.join(args)}):\n    if {c1}:\n        if {c2}:\n            print({args[0]})\n"


def t_ternary(rng):
    args = rng.sample(ARGS, 3)
    var = pick_var(rng)
    cond = rcond(rng, args, rng.randint(1, 3))
    a, b = rexpr(rng, args, rng.randint(0, 4)), rexpr(rng, args, rng.randint(0, 4))
    if rng.random() < 0.3:
        return f"def t({', '.join(args)}):\n    if {cond}:\n        return {a}\n    else:\n        return {b}\n"
    return (f"def t({', '.join(args)}):\n    if {cond}:\n        {var} = {a}\n    else:\n        {var} = {b}\n"
            f"    print({var})\n")


def t_redundant_else(rng):
    args = rng.sample(ARGS, 2)
    body = "\n".join(f"        print({rng.choice(args)})" for _ in range(rng.randint(2, 4)))
    return (f"def r({', '.join(args)}):\n    if {rcond(rng, args, 1)}:\n        return None\n"
            f"    else:\n{body}\n")


def t_unused_import(rng):
    mods = rng.sample(MODULES, rng.randint(2, 4))
    used = rng.sample(mods, rng.randint(0, len(mods) - 1))
    lines = [f"import {m}" for m in mods]
    if rng.random() < 0.4:
        lines.append(f"from {rng.choice(['os', 'math'])} import {rng.choice(['path', 'sqrt', 'pi'])}")
    lines += [f"print({m}.__name__)" for m in used]
    return "\n".join(lines) + "\n"


def t_dict_get(rng):
    d = rng.choice(["config", "cache", "counts", "settings", "lookup"])
    k = rng.choice(["key", "name", "k", "'debug'"])
    v = pick_var(rng)
    default = rng.choice(["0", "None", "''", "[]", "default", "compute_default()"])
    if rng.random() < 0.7:
        body = f"    if {k} in {d}:\n        {v} = {d}[{k}]\n    else:\n        {v} = {default}\n"
    else:
        body = f"    if {k} not in {d}:\n        {v} = {default}\n    else:\n        {v} = {d}[{k}]\n"
    return f"def lookup({d}, key, name, k, default):\n{body}    print({v})\n"


def t_not_compare(rng):
    a, b = rng.sample(ARGS, 2)
    form = rng.choice([f"not {a} == {b}", f"not {a} in {b}", f"not {a} is None", f"not not {a}", f"not {a} != {b}"])
    return f"def q({a}, {b}):\n    if {form}:\n        print({a})\n"


def t_unused_var(rng):
    args = rng.sample(ARGS, 2)
    var = pick_var(rng)
    value = rng.choice([str(rng.randint(0, 9)), "'cm'", "[]", rexpr(rng, args, 2), f"{args[0]}.attr", f"{args[0]}[0]"])
    return f"def u({', '.join(args)}):\n    {var} = {value}\n    return {args[0]} * {args[1]}\n"


def t_len_check(rng):
    seq = rng.choice(SEQS)
    cmp = rng.choice(["== 0", "> 0", "!= 0", ">= 1", "< 1"])
    return f"def e({seq}):\n    if len({seq}) {cmp}:\n        return None\n    return {seq}[0]\n"


def t_membership(rng):
    v = rng.choice(["day", "code", "x", "status", "color", "mode"])
    vals = rng.sample(["'a'", "'b'", "'c'", "1", "2", "3", "'red'", "'green'", "'blue'", "'Sat'", "'Sun'"],
                      rng.randint(2, 6))
    joiner, op = (" or ", "==") if rng.random() < 0.7 else (" and ", "!=")
    return f"def m({v}):\n    if {joiner.join(f'{v} {op} {x}' for x in vals)}:\n        return True\n    print({v})\n"


def t_chain(rng):
    a = rng.choice(["x", "age", "score", "i"])
    lo, hi = rng.randint(0, 10), rng.randint(20, 120)
    op1, op2 = rng.choice(["<", "<="]), rng.choice(["<", "<="])
    return f"def c({a}):\n    if {lo} {op1} {a} and {a} {op2} {hi}:\n        print({a})\n"


def t_pass(rng):
    seq = rng.choice(SEQS)
    return f"def p({seq}):\n    for x in {seq}:\n        pass\n        print(x)\n"


TEMPLATES = [t_temp_return, t_bool_return, t_singleton, t_comprehension, t_sum, t_range_len, t_augassign,
             t_constfold, t_dead_code, t_nested_if, t_ternary, t_redundant_else, t_unused_import, t_dict_get,
             t_not_compare, t_unused_var, t_len_check, t_membership, t_chain, t_pass]


# --- expert labelling policy ----------------------------------------------

def expert_label(f: dict) -> int:
    """Would a careful reviewer accept this refactoring automatically?"""
    rid = f["rule_id"]
    if f["risk"] >= 0.5:
        return 0                                   # may change behaviour
    if f["max_line_len_after"] > 88:
        return 0                                   # result too long (PEP 8 / black)
    if rid == "R01" and f["name_len"] >= 8 and f["nodes_before"] >= 12:
        return 0                                   # descriptive "explaining variable" for a complex expr
    if rid in ("R04", "R05"):
        if f["comp_generators"] >= 3 or (f["comp_generators"] == 2 and f["comp_has_if"]):
            return 0                               # nested comprehensions hurt readability
        if f["expr_depth_after"] >= 9 or f["replacement_len"] > 79:
            return 0
    if rid == "R08" and (f["replacement_len"] > 12 or f["nodes_before"] >= 5):
        return 0                                   # ugly float / self-documenting arithmetic
    if rid == "R11" and (f["replacement_len"] > 70 or f["bool_ops_after"] >= 3 or f["expr_depth_after"] >= 6):
        return 0                                   # unreadable conditional expression
    if rid == "R10" and f["replacement_len"] > 72:
        return 0
    if rid == "R02" and f["replacement_len"] > 79:
        return 0
    if rid == "R19" and f["replacement_len"] > 79:
        return 0
    return 1


def build_dataset(samples_per_template: int = 160, seed: int = 7, noise: float = 0.03) -> pd.DataFrame:
    rng = random.Random(seed)
    analyzer = Analyzer(ALL_RULES)
    rows = []
    for template in TEMPLATES:
        for _ in range(samples_per_template):
            code = template(rng)
            parsed = parse_source(code)
            if not parsed.ok:
                continue
            for match in analyzer.analyze(parsed.tree, code):
                feats = extract_features(match)
                label = expert_label(feats)
                if rng.random() < noise:
                    label = 1 - label
                feats["label"] = label
                feats["template"] = template.__name__
                rows.append(feats)
    df = pd.DataFrame(rows)
    return df[FEATURE_COLUMNS + ["label", "template"]]
