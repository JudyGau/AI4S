"""算子/终结符符号库与结构约束。

对齐 SymbArena（arXiv:2508.09897）：定义一元/二元算子与终结符，
并记录每类算子的合法子节点数，从源头避免非法数学表达式（非对称括号、负对数等）。
"""
from __future__ import annotations

import math
import random

import sympy as sp


# ---------------------------- 一元算子 ----------------------------
# 返回 (函数, 名称)。log 用自然对数，并对负域做保护。
UNARY_OPS: dict[str, ...] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "abs": sp.Abs,
    "sqrt": sp.sqrt,
}


# ---------------------------- 二元算子 ----------------------------
# 二元运算：+ - * / ^。sub/div 通过 Add/Mul 的系数技巧表示，见 expressions.py。
BINARY_OPS: list[str] = ["add", "sub", "mul", "div", "pow"]


# 结构约束：每个算子的合法子节点数量（叶子终结符计为 0）
ARITY = {name: 1 for name in UNARY_OPS}
ARITY.update({"add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2})


def make_symbols(dim: int, base: str = "x") -> list[sp.Symbol]:
    """生成自变量符号。dim=1 时为 x，否则为 x1..x_dim。"""
    if dim < 1:
        raise ValueError("dim 至少为 1")
    if dim == 1:
        return [sp.Symbol(base)]
    return [sp.Symbol(f"{base}{i + 1}") for i in range(dim)]


def random_literal_const(rng: random.Random) -> sp.Expr:
    """生成一个数值常数：保证非零、量级合理、有限小数。"""
    scale = rng.choice([1.0, 2.0, 3.0, 5.0])
    value = round(rng.uniform(0.2, 1.0) * scale, 2)
    value *= rng.choice([-1.0, 1.0])
    return sp.Float(value)


def random_integer(rng: random.Random, lo: int = 2, hi: int = 4) -> sp.Integer:
    return sp.Integer(rng.randint(lo, hi))