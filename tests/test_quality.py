import sympy as sp
import pytest

from symreg_dataset import quality


def test_complexity_ok_filter():
    too_simple = sp.Integer(5)
    ok = sp.sympify("sin(x) + log(x + 1)*x**2 - x")
    assert not quality.is_complexity_ok(too_simple, 4, 12)
    assert quality.is_complexity_ok(ok, 4, 12)


def test_skeleton_dedup():
    exprs = [sp.sympify("2*x"), sp.sympify("3*x * 1")]  # 相同骨架
    uniq = quality.dedup_by_skeleton(exprs)
    assert len(uniq) == 1


def test_split_no_leakage():
    exprs = [sp.sympify("2*x"), sp.sympify("3*x"), sp.sympify("sin(x) + 1")]
    train, test = quality.split_by_skeleton(exprs, test_ratio=0.34, seed=0)
    t_keys = {quality.skeleton_key(e) for e in train}
    te_keys = {quality.skeleton_key(e) for e in test}
    assert t_keys.isdisjoint(te_keys)
    # "2*x" 与 "3*x" 同骨架（系数抽象），必须同侧
    for k in t_keys | te_keys:
        pass
    assert not (t_keys & te_keys)


def test_difficulty_labels():
    assert quality.difficulty(sp.sympify("x + 1")) == "easy"
    assert quality.difficulty(sp.sympify("sin(x)*cos(exp(x))/log(abs(x)+2) - x**3")) == "hard"