"""Twenty AST refactoring patterns. Rules propose replacements; they do not apply them."""

from __future__ import annotations

import ast
import copy
import math
from typing import Iterable

from .analysis import ExprSlot, Match, NodeContext, ScopeInfo, StmtSpan
from .ast_tools import unparse

# Helpers

INVERSE_CMP = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.Lt: ast.GtE, ast.GtE: ast.Lt,
    ast.Gt: ast.LtE, ast.LtE: ast.Gt,
}
ORDER_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
BOOL_BUILTINS = {"isinstance", "issubclass", "callable", "hasattr", "all", "any", "bool"}
STR_PREDICATES = {"startswith", "endswith", "isdigit", "isalpha", "isalnum", "isspace", "islower",
                  "isupper", "istitle", "isnumeric", "isdecimal", "isidentifier", "isascii"}
SIDE_EFFECT_NODES = (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)


def dump(node: ast.AST) -> str:
    return ast.dump(node)


def same(a: ast.AST, b: ast.AST) -> bool:
    return dump(a) == dump(b)


def name(id_: str, store: bool = False) -> ast.Name:
    return ast.Name(id=id_, ctx=ast.Store() if store else ast.Load())


def assigned_name(node: ast.AST) -> str | None:
    """The variable in a simple `variable = expression` statement, if any."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def is_const(node: ast.AST, *types: type) -> bool:
    return isinstance(node, ast.Constant) and (not types or type(node.value) in types)


def is_bool_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, bool)


def is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def is_call_to(node: ast.AST, func: str, nargs: int | None = None) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func
            and not node.keywords and (nargs is None or len(node.args) == nargs)
            and not any(isinstance(a, ast.Starred) for a in node.args))


def is_simple_ref(node: ast.AST) -> bool:
    """``name`` or ``name.attr.attr`` - cheap, side-effect free to re-evaluate."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def has_side_effects(node: ast.AST) -> bool:
    return any(isinstance(n, SIDE_EFFECT_NODES) for n in ast.walk(node))


def name_loads(nodes: Iterable[ast.AST], ident: str) -> int:
    return sum(isinstance(n, ast.Name) and n.id == ident and isinstance(n.ctx, ast.Load)
               for root in nodes for n in ast.walk(root))


def names_mentioned(nodes: Iterable[ast.AST]) -> set[str]:
    return {n.id for root in nodes for n in ast.walk(root) if isinstance(n, ast.Name)}


def target_names(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def builtin_shadowed(ctx: NodeContext, ident: str, node: ast.AST) -> bool:
    scope: ScopeInfo | None = ctx.scope_for(node)
    while scope is not None:
        if scope.stores[ident]:
            return True
        scope = scope.parent
    return False


def is_boolean_expr(node: ast.AST) -> bool:
    """Does this expression always evaluate to a ``bool``?"""
    if isinstance(node, ast.Compare) or is_bool_const(node):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.BoolOp):
        return all(is_boolean_expr(v) for v in node.values)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in BOOL_BUILTINS:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in STR_PREDICATES:
            return True
    return False


def negate(expr: ast.expr) -> ast.expr:
    """Logical negation, simplified where it is exactly equivalent."""
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and type(expr.ops[0]) in (
            ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Is, ast.IsNot):
        return ast.Compare(left=expr.left, ops=[INVERSE_CMP[type(expr.ops[0])]()], comparators=expr.comparators)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return as_bool(expr.operand)
    return ast.UnaryOp(op=ast.Not(), operand=expr)


def as_bool(expr: ast.expr) -> ast.expr:
    return expr if is_boolean_expr(expr) else ast.Call(func=name("bool"), args=[expr], keywords=[])


def stmt_location(ctx: NodeContext, node: ast.AST | None = None) -> StmtSpan | None:
    entry = ctx.parent_entry(node)
    if entry is None or entry.index is None:
        return None
    container = getattr(entry.parent, entry.field)
    return StmtSpan(container, entry.index, entry.index + 1)


def expr_location(ctx: NodeContext, node: ast.AST | None = None) -> ExprSlot | None:
    entry = ctx.parent_entry(node)
    if entry is None:
        return None
    return ExprSlot(entry.parent, entry.field, entry.index)


def unroll_loop(loop: ast.For) -> tuple[list[ast.comprehension], ast.stmt] | None:
    """Turn ``for a in x: [for b in y:] [if c:] <stmt>`` into comprehension
    generators plus the single innermost statement."""
    gens: list[ast.comprehension] = []
    cur: ast.stmt = loop
    while True:
        if isinstance(cur, ast.For) and not cur.orelse and len(cur.body) == 1:
            gens.append(ast.comprehension(target=cur.target, iter=cur.iter, ifs=[], is_async=0))
            cur = cur.body[0]
        elif isinstance(cur, ast.If) and not cur.orelse and len(cur.body) == 1 and gens:
            gens[-1].ifs.append(cur.test)
            cur = cur.body[0]
        else:
            break
    if not gens:
        return None
    return gens, cur


def loop_var_leak_risk(ctx: NodeContext, loop: ast.For, gens: list[ast.comprehension]) -> float:
    """A ``for`` loop leaves its variable bound after the loop; a comprehension
    does not. If the variable is read outside the loop, the rewrite is risky."""
    scope = ctx.scope_for(loop)
    for gen in gens:
        for ident in target_names(gen.target):
            if ident in scope.declared:
                return 1.0
            if scope.loads[ident] > name_loads([loop], ident):
                return 0.6
    return 0.0


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return ""


def fresh_name(preferred: list[str], taken: set[str]) -> str:
    for cand in preferred:
        if cand and cand.isidentifier() and cand not in taken:
            return cand
    i = 1
    while f"item{i}" in taken:
        i += 1
    return f"item{i}"


# Rule base class

class Rule:
    rule_id = "R00"
    name = "Base rule"
    category = "General"
    summary = ""
    example_before = ""
    example_after = ""

    def match_node(self, node: ast.AST, ctx: NodeContext) -> Iterable[Match]:
        return ()

    def match_block(self, stmts: list, owner: ast.AST, field: str, ctx: NodeContext) -> Iterable[Match]:
        return ()

    def node_match(self, ctx: NodeContext, replacement, **details) -> Iterable[Match]:
        """Locate a single-node rewrite; block rules provide their own spans."""
        location = stmt_location(ctx) if isinstance(replacement, list) else expr_location(ctx)
        if location is not None:
            yield self.make(ctx, location, replacement, **details)

    def make(self, ctx: NodeContext, location, replacement, title: str, explanation: str,
             risk: float, **extra) -> Match:
        anchor = location.nodes()[0] if isinstance(location, StmtSpan) else location.node()
        return Match(rule=self, location=location, replacement=replacement, title=title,
                     explanation=explanation, risk=risk, ancestors=ctx.ancestor_statements(anchor),
                     context=ctx, extra=extra)


# Rules

class RedundantTempReturn(Rule):
    rule_id = "R01"
    name = "Redundant temporary variable"
    category = "Redundancy"
    summary = "A variable is assigned and then immediately returned."
    example_before = "def add(a, b):\n    result = a + b\n    return result"
    example_after = "def add(a, b):\n    return a + b"

    def match_block(self, stmts, owner, field, ctx):
        for i, (a, b) in enumerate(zip(stmts, stmts[1:])):
            var = assigned_name(a)
            if var is None:
                continue
            if not (isinstance(b, ast.Return) and isinstance(b.value, ast.Name) and b.value.id == var):
                continue
            scope = ctx.scope_for(a)
            if not scope.is_function or var in scope.declared or var in scope.closure_reads:
                continue
            yield self.make(
                ctx, StmtSpan(stmts, i, i + 2), [ast.Return(value=a.value)],
                title=f"The variable '{var}' is unnecessary because it is used only for returning the value.",
                explanation=(f"`{var}` is assigned on one line and returned on the next without being used "
                             "anywhere else. Returning the expression directly removes a redundant "
                             "store/load pair and one line of code."),
                risk=0.0, name_len=len(var),
            )


class SimplifyBooleanReturn(Rule):
    rule_id = "R02"
    name = "Boolean return simplification"
    category = "Simplification"
    summary = "`if cond: return True else: return False` returns the condition itself."
    example_before = "def is_adult(age):\n    if age >= 18:\n        return True\n    else:\n        return False"
    example_after = "def is_adult(age):\n    return age >= 18"

    @staticmethod
    def _bool_return(stmts) -> bool | None:
        if len(stmts) == 1 and isinstance(stmts[0], ast.Return) and stmts[0].value is not None \
                and is_bool_const(stmts[0].value):
            return stmts[0].value.value
        return None

    def match_block(self, stmts, owner, field, ctx):
        for i, s in enumerate(stmts):
            if not isinstance(s, ast.If):
                continue
            first = self._bool_return(s.body)
            if first is None:
                continue
            if s.orelse:
                second, end = self._bool_return(s.orelse), i + 1
            elif i + 1 < len(stmts):
                second, end = self._bool_return([stmts[i + 1]]), i + 2
            else:
                continue
            if second is None or second == first:
                continue
            value = as_bool(s.test) if first else negate(s.test)
            if builtin_shadowed(ctx, "bool", s) and "bool(" in unparse(value):
                continue
            yield self.make(
                ctx, StmtSpan(stmts, i, end), [ast.Return(value=value)],
                title="The if/else only returns True or False - return the condition directly.",
                explanation=("Branching just to return a boolean literal duplicates the condition's own value. "
                             "Returning the (possibly negated) condition is shorter and removes a branch. "
                             "If the condition is not already a bool it is wrapped in `bool()` so the result type "
                             "is unchanged."),
                risk=0.0,
            )


class CompareToSingleton(Rule):
    rule_id = "R03"
    name = "Comparison with None/True/False"
    category = "Idiomatic Python"
    summary = "Use `is None` instead of `== None`, and plain truthiness instead of `== True`."
    example_before = "if value == None:\n    ...\nif done == True:\n    ..."
    example_after = "if value is None:\n    ...\nif done:\n    ..."

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.Eq, ast.NotEq))):
            return
        left, right, eq = node.left, node.comparators[0], isinstance(node.ops[0], ast.Eq)
        if is_none(right) or is_none(left):
            other = left if is_none(right) else right
            if is_none(other):
                return
            new = ast.Compare(left=other, ops=[ast.Is() if eq else ast.IsNot()], comparators=[ast.Constant(None)])
            yield from self.node_match(
                ctx, new, title="Comparison to None should use `is` / `is not`.",
                explanation=("`None` is a singleton, so identity (`is`) is the correct and faster test. "
                             "`==` can be overridden by `__eq__` and give surprising results (PEP 8)."),
                risk=0.1,
            )
        elif (is_bool_const(right) or is_bool_const(left)) and ctx.in_bool_context(node):
            const, other = (right, left) if is_bool_const(right) else (left, right)
            if is_bool_const(other):
                return
            positive = (const.value is True) == eq
            new = other if positive else ast.UnaryOp(op=ast.Not(), operand=other)
            yield from self.node_match(
                ctx, new, title=f"Explicit comparison with {const.value} in a condition.",
                explanation=("Conditions are already evaluated for truthiness, so comparing with a boolean "
                             "literal is redundant (PEP 8). Note: `x == True` is only identical to `x` when "
                             "`x` is a real bool."),
                risk=0.05 if is_boolean_expr(other) else 0.35,
            )


class LoopToComprehension(Rule):
    rule_id = "R04"
    name = "Loop to comprehension"
    category = "Idiomatic Python"
    summary = "A loop that only builds a list/set/dict can be a comprehension."
    example_before = "squares = []\nfor n in numbers:\n    if n > 0:\n        squares.append(n * n)"
    example_after = "squares = [n * n for n in numbers if n > 0]"

    @staticmethod
    def _kind(value: ast.AST) -> str | None:
        if isinstance(value, ast.List) and not value.elts:
            return "list"
        if isinstance(value, ast.Dict) and not value.keys:
            return "dict"
        for kind in ("list", "dict", "set"):
            if is_call_to(value, kind, 0):
                return kind
        return None

    def match_block(self, stmts, owner, field, ctx):
        for i, (init, loop) in enumerate(zip(stmts, stmts[1:])):
            var = assigned_name(init)
            if var is None or not isinstance(loop, ast.For):
                continue
            kind = self._kind(init.value)
            unrolled = unroll_loop(loop) if kind else None
            if unrolled is None:
                continue
            gens, leaf = unrolled
            comp = self._build(kind, var, gens, leaf)
            if comp is None or builtin_shadowed(ctx, kind, init) and not isinstance(init.value, (ast.List, ast.Dict)):
                continue
            scope = ctx.scope_for(init)
            if isinstance(scope.node, ast.ClassDef) or var in names_mentioned([comp]):
                continue
            if any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.NamedExpr)) for n in ast.walk(comp)):
                continue
            risk = loop_var_leak_risk(ctx, loop, gens)
            label = {"list": "list", "set": "set", "dict": "dictionary"}[kind]
            explanation = (f"The loop exists only to fill `{var}`. A {label} comprehension expresses this "
                           "in one declarative expression and runs faster because the append/assignment "
                           "happens inside the interpreter's comprehension loop.")
            if risk:
                explanation += (" Caution: the loop variable is used after the loop; a comprehension does "
                                "not leak its variable, so this rewrite may change behaviour.")
            yield self.make(
                ctx, StmtSpan(stmts, i, i + 2), [ast.Assign(targets=[name(var, True)], value=comp)],
                title=f"The loop building `{var}` can be written as a {label} comprehension.",
                explanation=explanation, risk=risk,
                comp_generators=len(gens), comp_has_if=int(any(g.ifs for g in gens)),
            )

    @staticmethod
    def _build(kind, var, gens, leaf):
        if kind in ("list", "set"):
            method = "append" if kind == "list" else "add"
            if (isinstance(leaf, ast.Expr) and isinstance(leaf.value, ast.Call)
                    and isinstance(leaf.value.func, ast.Attribute) and leaf.value.func.attr == method
                    and isinstance(leaf.value.func.value, ast.Name) and leaf.value.func.value.id == var
                    and len(leaf.value.args) == 1 and not leaf.value.keywords
                    and not isinstance(leaf.value.args[0], ast.Starred)):
                cls = ast.ListComp if kind == "list" else ast.SetComp
                return cls(elt=leaf.value.args[0], generators=gens)
        elif kind == "dict":
            if (isinstance(leaf, ast.Assign) and len(leaf.targets) == 1
                    and isinstance(leaf.targets[0], ast.Subscript)
                    and isinstance(leaf.targets[0].value, ast.Name) and leaf.targets[0].value.id == var
                    and not isinstance(leaf.targets[0].slice, ast.Slice)):
                return ast.DictComp(key=leaf.targets[0].slice, value=leaf.value, generators=gens)
        return None


class AccumulatorToSum(Rule):
    rule_id = "R05"
    name = "Accumulator loop to sum()"
    category = "Idiomatic Python"
    summary = "A loop that only adds values to a counter is a call to `sum()`."
    example_before = "total = 0\nfor price in prices:\n    total += price"
    example_after = "total = sum(prices)"

    def match_block(self, stmts, owner, field, ctx):
        for i, (init, loop) in enumerate(zip(stmts, stmts[1:])):
            var = assigned_name(init)
            if not (var is not None and isinstance(loop, ast.For)
                    and is_const(init.value, int, float) and init.value.value == 0):
                continue
            unrolled = unroll_loop(loop)
            if unrolled is None:
                continue
            gens, leaf = unrolled
            if not (isinstance(leaf, ast.AugAssign) and isinstance(leaf.op, ast.Add)
                    and isinstance(leaf.target, ast.Name) and leaf.target.id == var):
                continue
            scope = ctx.scope_for(init)
            if isinstance(scope.node, ast.ClassDef) or builtin_shadowed(ctx, "sum", init):
                continue
            if var in names_mentioned([leaf.value] + [g.iter for g in gens] + [c for g in gens for c in g.ifs]):
                continue
            if any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.NamedExpr)) for n in ast.walk(leaf.value)):
                continue
            if (len(gens) == 1 and not gens[0].ifs and isinstance(gens[0].target, ast.Name)
                    and isinstance(leaf.value, ast.Name) and leaf.value.id == gens[0].target.id):
                arg = gens[0].iter
            else:
                arg = ast.GeneratorExp(elt=leaf.value, generators=gens)
            args = [arg] + ([ast.Constant(0.0)] if isinstance(init.value.value, float) else [])
            risk = loop_var_leak_risk(ctx, loop, gens)
            yield self.make(
                ctx, StmtSpan(stmts, i, i + 2),
                [ast.Assign(targets=[name(var, True)], value=ast.Call(func=name("sum"), args=args, keywords=[]))],
                title=f"The loop only accumulates into `{var}` - use the built-in sum().",
                explanation=("Initialising a counter to zero and adding to it in a loop is exactly what "
                             "`sum()` does. The built-in is implemented in C, is faster, and states the "
                             "intent in one line."),
                risk=risk, comp_generators=len(gens), comp_has_if=int(any(g.ifs for g in gens)),
            )


class _SubscriptReplacer(ast.NodeTransformer):
    def __init__(self, seq_dump: str, index: str, item: str):
        self.seq_dump, self.index, self.item = seq_dump, index, item

    def visit_Subscript(self, node):
        if (isinstance(node.ctx, ast.Load) and isinstance(node.slice, ast.Name)
                and node.slice.id == self.index and dump(node.value) == self.seq_dump):
            return ast.copy_location(name(self.item), node)
        return self.generic_visit(node)


class RangeLenLoop(Rule):
    rule_id = "R06"
    name = "range(len()) loop"
    category = "Idiomatic Python"
    summary = "Iterate over the sequence (or use enumerate) instead of indexing with range(len(...))."
    example_before = "for i in range(len(names)):\n    print(names[i])"
    example_after = "for name in names:\n    print(name)"

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and is_call_to(node.iter, "range", 1) and is_call_to(node.iter.args[0], "len", 1)):
            return
        seq = node.iter.args[0].args[0]
        if not is_simple_ref(seq) or builtin_shadowed(ctx, "range", node) or builtin_shadowed(ctx, "len", node):
            return
        idx, seq_dump = node.target.id, dump(seq)
        root = seq
        while isinstance(root, ast.Attribute):
            root = root.value
        element_loads = 0
        for n in (n for s in node.body for n in ast.walk(s)):
            if isinstance(n, ast.Name) and n.id in (idx, root.id) and not isinstance(n.ctx, ast.Load):
                return  # index or sequence re-bound inside the loop
            if isinstance(n, ast.Subscript) and dump(n.value) == seq_dump:
                if not isinstance(n.ctx, ast.Load):
                    return  # sequence is mutated through the index
                if isinstance(n.slice, ast.Name) and n.slice.id == idx:
                    element_loads += 1
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and dump(n.func.value) == seq_dump:
                return  # e.g. seq.append(...) changes the sequence while iterating
            if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
                return
        if not element_loads:
            return
        scope = ctx.scope_for(node)
        taken = scope.all_names() | names_mentioned([node])
        item = fresh_name([singular(getattr(seq, "id", getattr(seq, "attr", ""))), "item", "element", "value"], taken)
        new_loop = _SubscriptReplacer(seq_dump, idx, item).visit(copy.deepcopy(node))
        index_still_used = name_loads(new_loop.body, idx) > 0 or scope.loads[idx] > name_loads([node], idx)
        if index_still_used:
            if builtin_shadowed(ctx, "enumerate", node):
                return
            new_loop.target = ast.Tuple(elts=[name(idx, True), name(item, True)], ctx=ast.Store())
            new_loop.iter = ast.Call(func=name("enumerate"), args=[seq], keywords=[])
            how = f"`enumerate({unparse(seq)})` gives both the index and the element"
        else:
            new_loop.target = name(item, True)
            new_loop.iter = seq
            how = f"the index is only used to read `{unparse(seq)}[{idx}]`, so iterate over the elements directly"
        yield from self.node_match(
            ctx, [new_loop],
            title=f"Loop over `range(len({unparse(seq)}))` with indexing.",
            explanation=(f"Indexing a sequence inside a `range(len(...))` loop is un-Pythonic and error-prone; "
                         f"{how}."),
            risk=0.05,
        )


AUG_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift,
           ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd)


class AugmentedAssignment(Rule):
    rule_id = "R07"
    name = "Augmented assignment"
    category = "Simplification"
    summary = "`x = x + y` can be written `x += y`."
    example_before = "count = count + 1"
    example_after = "count += 1"

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, AUG_OPS)):
            return
        target = node.targets[0]
        if not (isinstance(target, (ast.Name, ast.Attribute, ast.Subscript)) and is_simple_ref(
                target.value if isinstance(target, ast.Subscript) else target)):
            return
        if isinstance(target, ast.Subscript) and has_side_effects(target.slice):
            return
        load_target = copy.deepcopy(target)
        for n in ast.walk(load_target):
            if hasattr(n, "ctx"):
                n.ctx = ast.Load()
        if not same(load_target, node.value.left):
            return
        right = node.value.right
        if is_const(right, int, float):
            risk = 0.0
        elif isinstance(node.value.op, (ast.Add, ast.Mult, ast.BitOr)):
            risk = 0.3  # in-place operators mutate lists/sets/dicts shared through aliases
        else:
            risk = 0.1
        op_sym = unparse(ast.AugAssign(target=name("x", True), op=node.value.op, value=ast.Constant(1)))[2:-2]
        yield from self.node_match(
            ctx, [ast.AugAssign(target=target, op=node.value.op, value=right)],
            title=f"`{unparse(target)}` is updated with its own value - use `{op_sym}`.",
            explanation=("Augmented assignment avoids repeating the target, evaluates it only once and "
                         "reads as an update. For mutable objects (lists, sets) `+=` works in place, so "
                         "check that no other reference relies on a copy."),
            risk=risk,
        )


_FOLD_OPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b, ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b,
    ast.BitOr: lambda a, b: a | b, ast.BitAnd: lambda a, b: a & b, ast.BitXor: lambda a, b: a ^ b,
}


def fold_constant(node: ast.AST):
    """Evaluate a constant expression at "compile time". Returns (ok, value)."""
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float, str):
            return True, node.value
        return False, None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
        ok, v = fold_constant(node.operand)
        if not ok or isinstance(v, str) or (isinstance(node.op, ast.Invert) and isinstance(v, float)):
            return False, None
        return True, {ast.USub: lambda x: -x, ast.UAdd: lambda x: +x, ast.Invert: lambda x: ~x}[type(node.op)](v)
    if isinstance(node, ast.BinOp) and type(node.op) in _FOLD_OPS:
        ok1, a = fold_constant(node.left)
        ok2, b = fold_constant(node.right)
        if not (ok1 and ok2):
            return False, None
        if isinstance(node.op, ast.Pow) and (not isinstance(b, (int, float)) or abs(b) > 64 or abs(a) > 10 ** 6):
            return False, None
        if isinstance(node.op, (ast.LShift, ast.RShift)) and isinstance(b, int) and b > 64:
            return False, None
        if isinstance(node.op, ast.Mult) and (isinstance(a, str) or isinstance(b, str)):
            count = b if isinstance(a, str) else a
            if not isinstance(count, int) or count > 40:
                return False, None
        try:
            value = _FOLD_OPS[type(node.op)](a, b)
        except Exception:  # ZeroDivisionError, TypeError, OverflowError ...
            return False, None
        if isinstance(value, complex) or (isinstance(value, float) and not math.isfinite(value)):
            return False, None
        if isinstance(value, int) and value.bit_length() > 64:
            return False, None
        if isinstance(value, str) and len(value) > 60:
            return False, None
        return True, value
    return False, None


class ConstantFolding(Rule):
    rule_id = "R08"
    name = "Constant folding"
    category = "Code Optimization"
    summary = "Expressions made only of literals are computed once, at refactoring time."
    example_before = "area = 3 * 4\ngreeting = 'Hello, ' + 'World'"
    example_after = "area = 12\ngreeting = 'Hello, World'"

    def match_node(self, node, ctx):
        if not isinstance(node, (ast.BinOp, ast.UnaryOp)):
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
            return  # a negative literal such as -1 is already "folded"
        entry = ctx.parent_entry()
        if entry and isinstance(entry.parent, (ast.BinOp, ast.UnaryOp)) and fold_constant(entry.parent)[0]:
            return  # fold the largest constant sub-expression only
        ok, value = fold_constant(node)
        if not ok:
            return
        yield from self.node_match(
            ctx, ast.Constant(value),
            title=f"`{unparse(node)}` only involves constants and can be folded to `{value!r}`.",
            explanation=("Constant folding is a classic compiler optimisation: an expression whose operands "
                         "are all literals is evaluated once instead of at run time. (CPython's own optimiser "
                         "does this for simple cases; folding in the source also simplifies the AST.) Keep "
                         "the original if the arithmetic documents meaning, e.g. `60 * 60 * 24`."),
            risk=0.0,
        )


class UnreachableCode(Rule):
    rule_id = "R09"
    name = "Unreachable code"
    category = "Dead Code"
    summary = "Statements after return/raise/break/continue can never run."
    example_before = "def f(x):\n    return x * 2\n    print('done')"
    example_after = "def f(x):\n    return x * 2"

    def match_block(self, stmts, owner, field, ctx):
        for k, s in enumerate(stmts[:-1]):
            if isinstance(s, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                kw = type(s).__name__.lower()
                count = len(stmts) - k - 1
                yield self.make(
                    ctx, StmtSpan(stmts, k + 1, len(stmts)), [],
                    title=f"{count} statement(s) after `{kw}` can never execute.",
                    explanation=(f"Control flow leaves the block at `{kw}` (line {s.lineno}), so the following "
                                 "code is dead. Dead-code elimination removes it without changing behaviour."),
                    risk=0.0, removed_statements=count,
                )
                return


def _and_operands(expr: ast.expr) -> list[ast.expr]:
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
        return list(expr.values)
    return [expr]


class MergeNestedIf(Rule):
    rule_id = "R10"
    name = "Merge nested if"
    category = "Simplification"
    summary = "`if a:` containing only `if b:` (no else) is `if a and b:`."
    example_before = "if user:\n    if user.active:\n        notify(user)"
    example_after = "if user and user.active:\n    notify(user)"

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.If) and not node.orelse and len(node.body) == 1
                and isinstance(node.body[0], ast.If) and not node.body[0].orelse):
            return
        inner = node.body[0]
        test = ast.BoolOp(op=ast.And(), values=_and_operands(node.test) + _and_operands(inner.test))
        yield from self.node_match(
            ctx, [ast.If(test=test, body=inner.body, orelse=[])],
            title="Nested `if` statements without `else` can be merged with `and`.",
            explanation=("Two nested conditions with no alternative branch are equivalent to one condition "
                         "joined with `and` (which also short-circuits). This removes one level of "
                         "indentation."),
            risk=0.0,
        )


class ConditionalExpression(Rule):
    rule_id = "R11"
    name = "Conditional expression"
    category = "Simplification"
    summary = "An if/else that only assigns (or returns) one of two values is a conditional expression."
    example_before = "if score >= 50:\n    grade = 'pass'\nelse:\n    grade = 'fail'"
    example_after = "grade = 'pass' if score >= 50 else 'fail'"

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.If) and len(node.body) == 1 and len(node.orelse) == 1):
            return
        a, b = node.body[0], node.orelse[0]
        var = assigned_name(a)
        if var is not None and var == assigned_name(b):
            new = ast.Assign(targets=[name(var, True)], value=ast.IfExp(test=node.test, body=a.value, orelse=b.value))
            what = f"assign `{var}`"
        elif (isinstance(a, ast.Return) and isinstance(b, ast.Return) and a.value is not None
              and b.value is not None and not (is_bool_const(a.value) and is_bool_const(b.value))):
            new = ast.Return(value=ast.IfExp(test=node.test, body=a.value, orelse=b.value))
            what = "return a value"
        else:
            return
        yield from self.node_match(
            ctx, [new],
            title=f"Both branches only {what} - use a conditional expression.",
            explanation=("When each branch of an if/else does the same thing with a different value, "
                         "`a if cond else b` states it in one line. Keep the if/else if the expression "
                         "becomes long or nested."),
            risk=0.0,
        )


class RedundantElse(Rule):
    rule_id = "R12"
    name = "Redundant else after return"
    category = "Simplification"
    summary = "An `else` after a branch that always returns/raises is unnecessary."
    example_before = "if x < 0:\n    raise ValueError('negative')\nelse:\n    process(x)\n    save(x)"
    example_after = "if x < 0:\n    raise ValueError('negative')\nprocess(x)\nsave(x)"

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.If) and node.orelse and node.body
                and isinstance(node.body[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break))):
            return
        a, b = node.body, node.orelse
        if len(a) == 1 and len(b) == 1 and type(a[0]) is type(b[0]) and isinstance(a[0], (ast.Return, ast.Assign)):
            return  # handled better by R02 / R11
        kw = type(node.body[-1]).__name__.lower()
        yield from self.node_match(
            ctx, [ast.If(test=node.test, body=node.body, orelse=[])] + list(node.orelse),
            title=f"`else` is redundant because the `if` branch ends with `{kw}`.",
            explanation=("If the first branch always leaves the block, the code after it only runs when the "
                         "condition is false anyway. De-indenting the `else` body flattens the code "
                         "(the 'guard clause' style)."),
            risk=0.0,
        )


class UnusedImport(Rule):
    rule_id = "R13"
    name = "Unused import"
    category = "Dead Code"
    summary = "Imported names that are never used."
    example_before = "import os\nimport sys\nprint(sys.argv)"
    example_after = "import sys\nprint(sys.argv)"

    def match_node(self, node, ctx):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or not ctx.scope.is_module:
            return
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            return
        if any(a.name == "*" for a in node.names) or any(isinstance(x, ast.Try) for x in ctx.ancestors()):
            return
        used = ctx.state.module_loads | ctx.state.dunder_all
        unused = [a for a in node.names if (a.asname or a.name).split(".")[0] not in used]
        if not unused:
            return
        keep = [a for a in node.names if a not in unused]
        if not keep:
            replacement = []
        elif isinstance(node, ast.Import):
            replacement = [ast.Import(names=keep)]
        else:
            replacement = [ast.ImportFrom(module=node.module, names=keep, level=node.level)]
        names_txt = ", ".join(f"`{a.asname or a.name}`" for a in unused)
        yield from self.node_match(
            ctx, replacement,
            title=f"Imported {names_txt} never used.",
            explanation=("Unused imports slow down start-up and mislead readers about dependencies. "
                         "Keep them only if the module re-exports the name or relies on the import's "
                         "side effects."),
            risk=0.2,
        )


class DictGetDefault(Rule):
    rule_id = "R14"
    name = "dict.get() with default"
    category = "Idiomatic Python"
    summary = "`if key in d: v = d[key] else: v = default` is `v = d.get(key, default)`."
    example_before = "if key in config:\n    value = config[key]\nelse:\n    value = 10"
    example_after = "value = config.get(key, 10)"

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.If) and len(node.body) == 1 and len(node.orelse) == 1
                and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], (ast.In, ast.NotIn))):
            return
        key, mapping = node.test.left, node.test.comparators[0]
        if not ((is_simple_ref(key) or isinstance(key, ast.Constant)) and is_simple_ref(mapping)):
            return
        hit, miss = (node.body[0], node.orelse[0]) if isinstance(node.test.ops[0], ast.In) else (node.orelse[0], node.body[0])
        var = assigned_name(hit)
        if var is None or var != assigned_name(miss):
            return
        v = hit.value
        if not (isinstance(v, ast.Subscript) and same(v.value, mapping) and same(v.slice, key)):
            return
        default = miss.value
        args = [key] if is_none(default) else [key, default]
        yield from self.node_match(
            ctx,
            [ast.Assign(targets=[name(var, True)],
                        value=ast.Call(func=ast.Attribute(value=mapping, attr="get", ctx=ast.Load()), args=args, keywords=[]))],
            title=f"Membership test followed by lookup - use `{unparse(mapping)}.get()`.",
            explanation=("`dict.get(key, default)` performs the membership test and the lookup in a single "
                         "hash operation and expresses the fallback directly."),
            risk=0.4 if has_side_effects(default) else 0.15,
        )


class SimplifyNegation(Rule):
    rule_id = "R15"
    name = "Negated comparison"
    category = "Simplification"
    summary = "`not a == b` is `a != b`, `not x in y` is `x not in y`."
    example_before = "if not item in seen:\n    ..."
    example_after = "if item not in seen:\n    ..."

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
            return
        operand = node.operand
        if isinstance(operand, ast.Compare) and len(operand.ops) == 1 and type(operand.ops[0]) in (
                ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Is, ast.IsNot):
            new = negate(operand)
            risk = 0.05 if isinstance(operand.ops[0], (ast.Eq, ast.NotEq)) else 0.0
            title = f"`{unparse(node)}` can use the inverse operator."
        elif isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
            inner = operand.operand
            new = inner if ctx.in_bool_context(node) else as_bool(inner)
            risk = 0.0
            title = "Double negation `not not x`."
        else:
            return
        yield from self.node_match(
            ctx, new, title=title,
            explanation=("Python has dedicated negated operators (`!=`, `not in`, `is not`). Using them "
                         "avoids a separate `not` and reads like English."),
            risk=risk,
        )


class UnusedVariable(Rule):
    rule_id = "R16"
    name = "Unused local variable"
    category = "Dead Code"
    summary = "A local variable is assigned but never read."
    example_before = "def area(w, h):\n    unit = 'cm'\n    return w * h"
    example_after = "def area(w, h):\n    return w * h"

    def match_node(self, node, ctx):
        var = assigned_name(node)
        if var is None:
            return
        scope = ctx.scope
        if not scope.is_function or var.startswith("_") or scope.loads[var] or var in scope.declared:
            return
        if has_side_effects(node.value):
            return
        if any(is_call_to(n, f) for n in ast.walk(scope.node) for f in ("locals", "vars", "eval", "exec")):
            return
        literal_only = all(isinstance(n, (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Load))
                           for n in ast.walk(node.value))
        yield from self.node_match(
            ctx, [],
            title=f"Local variable `{var}` is assigned but never used.",
            explanation=("The value is computed and stored but nothing reads it - a dead store. Removing it "
                         "simplifies the function. (Only side-effect-free assignments are reported.)"),
            risk=0.05 if literal_only else 0.3, name_len=len(var),
        )


class EmptyCollectionCheck(Rule):
    rule_id = "R17"
    name = "Empty collection check"
    category = "Idiomatic Python"
    summary = "Use the truthiness of a collection instead of comparing len() with 0."
    example_before = "if len(items) == 0:\n    return None"
    example_after = "if not items:\n    return None"

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and is_call_to(node.left, "len", 1)
                and is_const(node.comparators[0], int) and ctx.in_bool_context(node)):
            return
        if builtin_shadowed(ctx, "len", node):
            return
        op, n, seq = type(node.ops[0]), node.comparators[0].value, node.left.args[0]
        if (op is ast.Eq and n == 0) or (op is ast.Lt and n == 1) or (op is ast.LtE and n == 0):
            new = ast.UnaryOp(op=ast.Not(), operand=seq)
        elif (op in (ast.NotEq, ast.Gt) and n == 0) or (op is ast.GtE and n == 1):
            new = seq
        else:
            return
        yield from self.node_match(
            ctx, new, title=f"`{unparse(node)}` can use the collection's truthiness.",
            explanation=("Empty sequences and collections are falsy (PEP 8), so `if not items:` is the "
                         "idiomatic emptiness test. Not valid for objects such as NumPy arrays."),
            risk=0.3,
        )


class RedundantPass(Rule):
    rule_id = "R18"
    name = "Redundant pass"
    category = "Dead Code"
    summary = "`pass` in a block that has other statements does nothing."
    example_before = "for x in data:\n    pass\n    process(x)"
    example_after = "for x in data:\n    process(x)"

    def match_block(self, stmts, owner, field, ctx):
        if len(stmts) < 2:
            return
        for k, s in enumerate(stmts):
            if isinstance(s, ast.Pass):
                yield self.make(
                    ctx, StmtSpan(stmts, k, k + 1), [],
                    title="`pass` is redundant in a non-empty block.",
                    explanation="`pass` is only needed where a block would otherwise be empty.",
                    risk=0.0,
                )


class MembershipTest(Rule):
    rule_id = "R19"
    name = "Multiple comparisons to membership test"
    category = "Simplification"
    summary = "`x == a or x == b or x == c` is `x in (a, b, c)`."
    example_before = "if day == 'Sat' or day == 'Sun':\n    rest()"
    example_after = "if day in ('Sat', 'Sun'):\n    rest()"

    def match_node(self, node, ctx):
        if not isinstance(node, ast.BoolOp) or len(node.values) < 2:
            return
        want, new_op = (ast.Eq, ast.In) if isinstance(node.op, ast.Or) else (ast.NotEq, ast.NotIn)
        subject, options = None, []
        for v in node.values:
            if not (isinstance(v, ast.Compare) and len(v.ops) == 1 and isinstance(v.ops[0], want)):
                return
            if subject is None:
                subject = v.left
                if not is_simple_ref(subject):
                    return
            elif not same(subject, v.left):
                return
            option = v.comparators[0]
            if not (isinstance(option, ast.Constant) or is_simple_ref(option)):
                return
            options.append(option)
        new = ast.Compare(left=subject, ops=[new_op()], comparators=[ast.Tuple(elts=options, ctx=ast.Load())])
        word = "in" if new_op is ast.In else "not in"
        yield from self.node_match(
            ctx, new,
            title=f"`{unparse(subject)}` is compared with {len(options)} values - use `{word}`.",
            explanation=("Repeating the same variable in a chain of `==`/`or` (or `!=`/`and`) comparisons "
                         "is verbose; a membership test lists the allowed values once and evaluates the "
                         "subject only once."),
            risk=0.05, comparisons=len(options),
        )


class ChainedComparison(Rule):
    rule_id = "R20"
    name = "Chained comparison"
    category = "Simplification"
    summary = "`a < b and b < c` is `a < b < c`."
    example_before = "if 0 <= age and age < 120:\n    ..."
    example_after = "if 0 <= age < 120:\n    ..."

    def match_node(self, node, ctx):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)):
            return
        values = node.values
        for i, (a, b) in enumerate(zip(values, values[1:])):
            if not (isinstance(a, ast.Compare) and isinstance(b, ast.Compare) and len(b.ops) == 1
                    and all(isinstance(o, ORDER_OPS) for o in a.ops + b.ops)):
                continue
            middle = a.comparators[-1]
            if not (same(middle, b.left) and (is_simple_ref(middle) or isinstance(middle, ast.Constant))):
                continue
            merged = ast.Compare(left=a.left, ops=a.ops + b.ops, comparators=a.comparators + b.comparators)
            rest = values[:i] + [merged] + values[i + 2:]
            new = merged if len(rest) == 1 else ast.BoolOp(op=ast.And(), values=rest)
            yield from self.node_match(
                ctx, new,
                title=f"Comparisons sharing `{unparse(middle)}` can be chained.",
                explanation=("Python supports mathematical comparison chains: `a < b < c` means "
                             "`a < b and b < c` but evaluates `b` once and reads like the range it describes."),
                risk=0.0,
            )
            return


ALL_RULES: list[Rule] = [
    RedundantTempReturn(), SimplifyBooleanReturn(), CompareToSingleton(), LoopToComprehension(),
    AccumulatorToSum(), RangeLenLoop(), AugmentedAssignment(), ConstantFolding(), UnreachableCode(),
    MergeNestedIf(), ConditionalExpression(), RedundantElse(), UnusedImport(), DictGetDefault(),
    SimplifyNegation(), UnusedVariable(), EmptyCollectionCheck(), RedundantPass(), MembershipTest(),
    ChainedComparison(),
]
RULES_BY_ID = {r.rule_id: r for r in ALL_RULES}


def get_rules(rule_ids: Iterable[str] | None = None) -> list[Rule]:
    if rule_ids is None:
        return list(ALL_RULES)
    wanted = set(rule_ids)
    return [r for r in ALL_RULES if r.rule_id in wanted]
