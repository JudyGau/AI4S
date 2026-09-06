"""多变量随机方程生成（树式）+ 参数/骨架抽取。

对齐 SymbArena（arXiv:2508.09897）：
- 用递归增量方式构造表达式树，叶子为自变量或常数；
- skeletonize 把数值常数替换为自由参数符号，得到「表达式骨架」，
  用作形式唯一性校验与形式奖励的基础。
"""
from __future__ import annotations

import math
import random

import sympy as sp

from .symbols import (
    UNARY_OPS,
    BINARY_OPS,
    make_symbols,
    random_literal_const,
    random_integer,
)


def _terminal(xs: list, rng: random.Random) -> sp.Expr:
    """叶子节点：自变量（约 60%）或常数。"""
    if xs and rng.random() < 0.6:
        return rng.choice(xs)
    return random_literal_const(rng)


def _apply_unary(name: str, arg: sp.Expr, rng: random.Random) -> sp.Expr:
    """应用一元算子，并对易出错的定义域做保护。"""
    f = UNARY_OPS[name]
    if name == "log":
        return sp.log(sp.Abs(arg))  # 避免负对数
    if name == "sqrt":
        return sp.sqrt(sp.Abs(arg))  # 避免负开根
    return f(arg)


def _apply_binary(name: str, a: sp.Expr, b: sp.Expr, rng: random.Random) -> sp.Expr:
    """应用二元算子。sub/div 借助 Add/Mul 表达以保持结构可解析。"""
    if name == "add":
        return a + b
    if name == "sub":
        return a - b
    if name == "mul":
        return a * b
    if name == "div":
        return a / b
    if name == "pow":
        # 指数取小整数，避免负底小数次方导致复数
        return a ** random_integer(rng, 2, 4)
    raise ValueError(name)


def random_expression(dim: int = 1, max_depth: int = 4, rng: random.Random | None = None,
                      var_symbols: list[sp.Symbol] | None = None,
                      unary_names: list[str] | None = None,
                      binary_names: list[str] | None = None) -> sp.Expr:
    """随机生成一/多元表达式。

    参数同 SymbArena：控制递归深度（决定复杂度）。通过 `p_term` 使叶子概率随深度升高，
    避免树无界增长。返回 SymPy 表达式。
    """
    if rng is None:
        rng = random.Random()
    if max_depth < 1:
        raise ValueError("max_depth 至少为 1")
    xs = var_symbols or make_symbols(dim)
    if unary_names is None:
        unary_names = list(UNARY_OPS)
    if binary_names is None:
        binary_names = list(BINARY_OPS)

    def _build(depth: int) -> sp.Expr:
        # 到达最大深度则强制叶节点
        if depth >= max_depth:
            return _terminal(xs, rng)
        # 叶子概率随深度增加，保证树大小受控
        p_term = 0.15 + 0.15 * depth / max(max_depth, 1)
        if rng.random() < p_term:
            return _terminal(xs, rng)
        if rng.random() < 0.5 or not unary_names:
            name = rng.choice(binary_names)
            a, b = _build(depth + 1), _build(depth + 1)
            return _apply_binary(name, a, b, rng)
        name = rng.choice(unary_names)
        return _apply_unary(name, _build(depth + 1), rng)

    return _build(0)


def skeletonize(expr: sp.Expr, base: str = "c") -> tuple[sp.Expr, dict]:
    """把表达式中的数值常数替换为自由参数符号，得「表达式骨架」。

    返回 (skeleton_symbolic, param_map)。param_map 将数值映射到参数符号。
    用 SymPy atoms 保证幂等的系数抽象，作为形式唯一性与形式奖励的依据。
    """
    numbers = [n for n in expr.atoms(sp.Number) if not bool(n.is_integer and str(n) in ('0', '1'))]
    # 确定性排序：避免同一表达式产生不同骨架
    order = sorted(numbers, key=lambda n: (float(n), str(n)))
    mapping: dict = {}
    for i, n in enumerate(order):
        p = sp.Symbol(f"{base}{i}")
        mapping[n] = p
        if n.is_integer and not mapping.get(n):
            pass
    # 0 和 1 往往是结构常量（+0 / *1），保留不动；其余系数抽象。
    keep = {sp.Integer(0), sp.Integer(1)}
    clean = {n: p for n, p in mapping.items() if n not in keep}
    skeleton = expr.xreplace(clean)
    return skeleton, clean