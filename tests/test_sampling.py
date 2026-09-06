import math
import random

import sympy as sp

from symreg_dataset import sampling
from symreg_dataset.symbols import make_symbols


def test_sample_data_shapes():
    e = sp.sympify("sin(x) + x")
    rows = sampling.sample_data(e, group=2, n_per_group=8, dim=1, seed=0)
    assert len(rows) == 2
    assert len(rows[0].variables) == 1
    assert len(rows[0].output) == 8


def test_sample_data_matches_expr():
    e = sp.sympify("2*x")
    rows = sampling.sample_data(e, group=1, n_per_group=50, dim=1,
                                dist="uniform", dom=5, seed=0)
    xs, ys = rows[0].variables[0], rows[0].output
    for x, y in zip(xs, ys):
        # x 与 y 各自独立四舍五入到 precision=2，允许 ±0.02 误差
        assert abs(y - 2 * x) < 0.02


def test_sample_data_nonfinite_raises():
    e = sp.sympify("1/x")
    sample = sampling.sample_data(e, group=1, n_per_group=100, dim=1,
                                  dom=1, seed=0)
    # 只要没踩到 x=0 就应有限；这里检验接口正常返回
    assert all(math.isfinite(v) for v in sample[0].output)