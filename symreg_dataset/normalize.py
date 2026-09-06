"""表达式规范化：canonical 化、系数抽象、稳定序列化。

对齐 SymbArena（arXiv:2508.09897）：形式一致性评借助「系数抽象后的规范形式」。
这里提供三个核心能力：
- serialize：稳定、可复现的字符串序列化（固定精度小数）；
- canonical：simplify + 项排序 + 系数抽象 → 规范形式字符串（骨架）；
- parse_expression_text：从可能带 `f(x)=` 前缀的文本中提取 SymPy 表达式。
"""
from __future__ import annotations

import re

import sympy as sp

from .expressions import skeletonize


def _round_constants(expr: sp.Expr, precision: int = 2) -> sp.Expr:
    # 整数（含结构性的 0/1）不动；其余数值常量四舍五入到 precision 位小数，保证可复现
    mapping = {}
    for n in expr.atoms(sp.Number):
        if n.is_Integer:
            continue
        mapping[n] = sp.Float(round(float(n), precision), precision + 2)
    return expr.xreplace(mapping)


def serialize(expr: sp.Expr, precision: int = 2) -> str:
    """稳定序列化：常数保留最多 precision 位小数。"""
    e = _round_constants(expr, precision)
    s = sp.sstr(e).replace(" ", "")
    return s


def canonical(expr: sp.Expr, precision: int = 2) -> str:
    """simplify + 项排序 + 系数抽象 → 规范形式字符串（即骨架的稳定表达）。"""
    e = sp.simplify(expr)
    skeleton, _ = skeletonize(e, base="c")
    # 确定性的项排序，避免顺序抖动
    s = str(sp.expand(skeleton))
    return s.replace(" ", "")


def parse_expression_text(text: str) -> sp.Expr:
    """从候选文本中解析 SymPy 表达式。

    支持去掉常见前缀，如 `f(x)=...`、`y = ...`，并容忍换行/中文引号。
    解析失败抛 ValueError。
    """
    t = text.strip().replace("，", ",")
    # 去掉中文/英文等号左侧的 `f(x)` / `y` 等
    t = re.sub(r"^[a-zA-Z_]+\([^)]*\)\s*=\s*", "", t)
    t = re.sub(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*", "", t)
    t = t.rstrip(".;,。")
    return sp.sympify(t)