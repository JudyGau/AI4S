"""通用工具函数。"""
from __future__ import annotations

import random

import sympy as sp

from .normalize import parse_expression_text, serialize


def setup_rng(seed: int | None = None, default: int = 0) -> random.Random:
    return random.Random(seed if seed is not None else default)


def round_list(vals: list[float], precision: int = 2) -> list[float]:
    return [round(float(v), precision) for v in vals]


def parse_candidate(text: str) -> sp.Expr:
    """解析模型候选表达式，失败抛 ValueError。"""
    return parse_expression_text(text)


def expr_text(expr: sp.Expr, precision: int = 2) -> str:
    """稳定的表达式字符串（用于 output / reference）。"""
    return serialize(expr, precision)