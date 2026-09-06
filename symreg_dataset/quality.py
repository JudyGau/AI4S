"""难度与质量控制。

对齐 SymbArena（arXiv:2508.09897）Step D：
- 复杂度过滤（树深 4~12，算子数/终结符数统计）；
- 骨架唯一性（coefficient-abstracted 规范形式去重）；
- 难度分级（easy / medium / hard）；
- 按骨架划分 train/test，防止同形式不同系数泄露。
"""
from __future__ import annotations

import random

import sympy as sp

from .normalize import canonical, serialize


# ---------------------------- 复杂度 ----------------------------
def expression_depth(expr: sp.Expr) -> int:
    """递归计算表达式树高度。"""
    if not expr.args:
        return 1
    return 1 + max(expression_depth(a) for a in expr.args)


def complexity(expr: sp.Expr) -> dict[str, int]:
    """复杂度统计：算子数、节点数、深度、变量数。"""
    return {
        "ops": expr.count_ops(),
        "nodes": expr.count_ops() + len(expr.atoms(sp.Symbol)) + len(expr.atoms(sp.Number)),
        "depth": expression_depth(expr),
        "vars": len(expr.free_symbols),
    }


def difficulty(expr: sp.Expr) -> str:
    """按复杂度分级。"""
    c = complexity(expr)
    ops, depth = c["ops"], c["depth"]
    if ops <= 3 and depth <= 5:
        return "easy"
    if ops <= 6 and depth <= 8:
        return "medium"
    return "hard"


def is_complexity_ok(expr: sp.Expr, min_depth: int = 4, max_depth: int = 12,
                     **_) -> bool:
    """按 SymbArena 用树深度过滤过度简单/过度复杂。"""
    c = complexity(expr)
    return min_depth <= c["depth"] <= max_depth


# ---------------------------- 唯一性 ----------------------------
def skeleton_key(expr: sp.Expr) -> str:
    """骨架规范形式作为结构级唯一键。"""
    return canonical(expr)


def dedup_by_skeleton(expressions: list[sp.Expr]) -> list[sp.Expr]:
    """按骨架去重，保证结构级唯一。"""
    seen: set[str] = set()
    out: list[sp.Expr] = []
    for e in expressions:
        k = skeleton_key(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def is_unique(expr: sp.Expr, seen: set[str]) -> bool:
    k = skeleton_key(expr)
    if k in seen:
        return False
    seen.add(k)
    return True


# ---------------------------- train/test 划分 ----------------------------
def split_by_skeleton(expressions: list[sp.Expr], test_ratio: float = 0.1,
                      seed: int | None = None) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """把表达式按骨架分成 train/test，保证同骨架样本只落在同一侧，防止形式泄露。"""
    rng = random.Random(seed)
    buckets: dict[str, list[sp.Expr]] = {}
    for e in expressions:
        buckets.setdefault(skeleton_key(e), []).append(e)
    keys = list(buckets)
    rng.shuffle(keys)
    n_test = max(1, int(len(keys) * test_ratio))
    test_keys = set(keys[:n_test])
    train = [e for k, ea in buckets.items() if k not in test_keys for e in ea]
    test = [e for k in test_keys for e in buckets[k]]
    return train, test