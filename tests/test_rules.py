"""Unit tests: each rule's transformation and its behaviour-preservation guards."""

import textwrap

import pytest

from refactor_engine import run_pipeline


def refactor(code: str, rule: str, preserve: bool = True) -> str:
    result = run_pipeline(textwrap.dedent(code), rule_ids=[rule], apply_mode="all", preserve_formatting=preserve)
    assert result.ok
    return result.refactored_source


def norm(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


CASES = [
    ("R01", """
        def add(a, b):
            result = a + b
            return result
     """, """
        def add(a, b):
            return a + b
     """),
    ("R02", """
        def is_adult(age):
            if age >= 18:
                return True
            else:
                return False
     """, """
        def is_adult(age):
            return age >= 18
     """),
    ("R02", """
        def empty(xs):
            if xs:
                return False
            return True
     """, """
        def empty(xs):
            return not xs
     """),
    ("R02", """
        def has(x):
            if x:
                return True
            return False
     """, """
        def has(x):
            return bool(x)
     """),
    ("R03", """
        if value == None:
            pass
     """, """
        if value is None:
            pass
     """),
    ("R03", """
        if done == False:
            pass
     """, """
        if not done:
            pass
     """),
    ("R04", """
        def f(numbers):
            squares = []
            for n in numbers:
                if n > 0:
                    squares.append(n * n)
            return squares
     """, """
        def f(numbers):
            squares = [n * n for n in numbers if n > 0]
            return squares
     """),
    ("R04", """
        def f(words):
            d = {}
            for w in words:
                d[w] = len(w)
            return d
     """, """
        def f(words):
            d = {w: len(w) for w in words}
            return d
     """),
    ("R05", """
        def f(prices):
            total = 0
            for p in prices:
                total += p
            return total
     """, """
        def f(prices):
            total = sum(prices)
            return total
     """),
    ("R05", """
        def f(xs):
            c = 0.0
            for x in xs:
                c += x * 2
            return c
     """, """
        def f(xs):
            c = sum((x * 2 for x in xs), 0.0)
            return c
     """),
    ("R06", """
        for i in range(len(names)):
            print(names[i])
     """, """
        for name in names:
            print(name)
     """),
    ("R06", """
        for i in range(len(names)):
            print(i, names[i])
     """, """
        for i, name in enumerate(names):
            print(i, name)
     """),
    ("R07", "count = count + 1\n", "count += 1\n"),
    ("R08", "area = 3 * 4\n", "area = 12\n"),
    ("R08", "x = y * (2 + 3)  # keep me\n", "x = y * (5)  # keep me\n"),
    ("R09", """
        def f(x):
            return x * 2
            print("done")
     """, """
        def f(x):
            return x * 2
     """),
    ("R10", """
        if user:
            if user.active:
                notify(user)
     """, """
        if user and user.active:
            notify(user)
     """),
    ("R11", """
        if score >= 50:
            grade = "pass"
        else:
            grade = "fail"
     """, """
        grade = "pass" if score >= 50 else "fail"
     """),
    ("R12", """
        def f(x):
            if x < 0:
                raise ValueError("neg")
            else:
                print(x)
                return x
     """, """
        def f(x):
            if x < 0:
                raise ValueError("neg")
            print(x)
            return x
     """),
    ("R13", """
        import os
        import sys
        print(sys.argv)
     """, """
        import sys
        print(sys.argv)
     """),
    ("R13", """
        from math import pi, sqrt
        print(sqrt(2))
     """, """
        from math import sqrt
        print(sqrt(2))
     """),
    ("R14", """
        if key in config:
            value = config[key]
        else:
            value = 10
     """, """
        value = config.get(key, 10)
     """),
    ("R15", """
        if not item in seen:
            pass
     """, """
        if item not in seen:
            pass
     """),
    ("R16", """
        def area(w, h):
            unit = "cm"
            return w * h
     """, """
        def area(w, h):
            return w * h
     """),
    ("R17", """
        if len(items) == 0:
            pass
     """, """
        if not items:
            pass
     """),
    ("R18", """
        for x in data:
            pass
            print(x)
     """, """
        for x in data:
            print(x)
     """),
    ("R19", """
        if day == "Sat" or day == "Sun":
            pass
     """, """
        if day in ("Sat", "Sun"):
            pass
     """),
    ("R20", """
        if 0 <= age and age < 120:
            pass
     """, """
        if 0 <= age < 120:
            pass
     """),
]


@pytest.mark.parametrize("rule,before,after", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_rule_transformation(rule, before, after):
    assert refactor(before, rule) == norm(after)


@pytest.mark.parametrize("rule,before,after", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_regenerate_mode_gives_same_ast(rule, before, after):
    import ast
    assert ast.dump(ast.parse(refactor(before, rule, preserve=False))) == ast.dump(ast.parse(norm(after)))


# --- behaviour-preservation guards: these must NOT be rewritten -------------

UNCHANGED = [
    ("R01", """
        def f():
            x = 1
            g = lambda: x
            x = 2
            return x
     """),  # x captured by a closure
    ("R01", """
        def f():
            global total
            total = 5
            return total
     """),
    ("R04", """
        class C:
            factor = 2
            out = []
            for n in range(3):
                out.append(n * factor)
     """),  # comprehensions cannot see class variables
    ("R04", """
        def f(xs):
            out = []
            for x in xs:
                out.append(x)
                print(x)
            return out
     """),  # loop does more than append
    ("R05", """
        def f(xs, sum):
            t = 0
            for x in xs:
                t += x
            return t
     """),  # builtin sum shadowed
    ("R06", """
        for i in range(len(xs)):
            xs[i] = xs[i] * 2
     """),  # writes through the index
    ("R06", """
        for i in range(len(xs)):
            xs.append(xs[i])
     """),  # mutates the sequence
    ("R08", "x = -1\n"),
    ("R08", "x = 1 / 0\n"),
    ("R08", "x = 2 ** 10000\n"),
    ("R13", """
        try:
            import numpy
        except ImportError:
            numpy = None
     """),
    ("R16", """
        def f():
            r = compute()
            return 1
     """),  # call may have side effects
    ("R17", "n = len(xs) == 0\n"),  # not in a boolean context
    ("R03", "same = flag == True\n"),  # not in a boolean context
]


@pytest.mark.parametrize("rule,code", UNCHANGED, ids=[f"{c[0]}-{i}" for i, c in enumerate(UNCHANGED)])
def test_unsafe_patterns_are_not_rewritten(rule, code):
    assert refactor(code, rule) == norm(code)


def test_loop_variable_used_after_loop_is_risky():
    code = """
        def f(xs):
            out = []
            for x in xs:
                out.append(x)
            print(x)
            return out
    """
    result = run_pipeline(textwrap.dedent(code), rule_ids=["R04"])
    (s,) = result.suggestions
    assert s.risk >= 0.5 and not s.will_apply
