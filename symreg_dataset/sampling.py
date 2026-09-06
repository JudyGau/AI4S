"""数据矩阵采样。

对齐 SymbArena（arXiv:2508.09897）：
- 每个方程配 data matrix d_i，含 group 组数据，每组由 dim 个自变量 + 1 个因变量构成；
- 自变量从 U(-dom, dom) 或 N(0, dom) 采样（dom 默认 10）；
- 可加高斯噪声；过滤 NaN/Inf 与数值不稳定取值。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from .symbols import make_symbols
from .normalize import _round_constants


@dataclass
class DataGroup:
    """一组数据：自变量矩阵（list of list）+ 因变量向量。"""
    variables: list[list[float]]
    output: list[float]
    dist: str = "uniform"
    dom: float = 10.0
    noise: float = 0.0


def _lambdify(expr: sp.Expr, var_symbols: list[sp.Symbol]):
    # 用 numpy，可稳健处理 re/im/cosh/sinh 等由 pow/除法/abs 派生的节点
    return sp.lambdify(var_symbols, expr, "numpy")


def _real(x):
    """提取数值的实部（numpy/scalar）；非数值返回 NaN。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        try:
            return float(np.real(x))
        except Exception:
            return float("nan")


def sample_data(expr: sp.Expr, group: int = 1, n_per_group: int = 32,
                dim: int = 1, dist: str = "uniform", dom: float = 10.0,
                noise: float = 0.0, seed: int | None = None,
                var_symbols: list[sp.Symbol] | None = None,
                precision: int = 2) -> list[DataGroup]:
    """为给定表达式采样数据矩阵。

    返回 List[DataGroup]，每组长度为 n_per_group。若任一组出现非有限值则抛 ValueError，
    由上层重试（起质量过滤作用）。
    """
    if dist not in ("uniform", "gaussian"):
        raise ValueError("dist 仅支持 uniform 或 gaussian")
    rng = random.Random(seed)
    xs = var_symbols or make_symbols(dim)
    fn = _lambdify(expr, xs)
    e = _round_constants(expr, precision)

    def _sample_var():
        if dist == "uniform":
            return rng.uniform(-dom, dom)
        return rng.gauss(0.0, dom)

    groups: list[DataGroup] = []
    for _ in range(group):
        variables: list[list[float]] = [[] for _ in range(dim)]
        output: list[float] = []
        for _ in range(n_per_group):
            point = [_sample_var() for _ in range(dim)]
            with np.errstate(all="ignore"):
                y = float(_real(fn(*point)))
            if noise > 0:
                y += rng.gauss(0.0, noise)
            if not math.isfinite(y) or any(not math.isfinite(v) for v in point):
                raise ValueError("采样得到非有限值，表达式数值不稳定")
            for d in range(dim):
                variables[d].append(round(point[d], precision))
            output.append(round(y, precision))
        groups.append(DataGroup(variables=variables, output=output,
                                dist=dist, dom=dom, noise=noise))
    return groups