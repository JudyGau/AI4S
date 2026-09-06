import random

import sympy as sp

from symreg_dataset import expressions, quality


def test_random_expression_dim():
    for dim in (1, 2, 3):
        e = expressions.random_expression(dim=dim, max_depth=3, rng=random.Random(0))
        xs = e.free_symbols
        assert len({str(s) for s in xs}) <= dim


def test_random_expression_depth_bounded():
    e = expressions.random_expression(dim=1, max_depth=4, rng=random.Random(1))
    assert quality.expression_depth(e) <= 4


def test_skeleton_replaces_constants():
    e = sp.sympify("2*x + 3")
    sk, mapping = expressions.skeletonize(e)
    assert any(str(s).startswith("c") for s in sk.free_symbols)


def test_skeleton_deterministic():
    e1 = sp.sympify("3*sin(x) + 0.5*x")
    e2 = sp.sympify("3*sin(x) + 0.5*x")
    assert expressions.skeletonize(e1)[1] == expressions.skeletonize(e2)[1]