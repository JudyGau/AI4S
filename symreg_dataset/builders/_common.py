"""builder 公共基础设施：校验表达式、采样数据、构造 reference 元数据。"""
from __future__ import annotations

import random

import sympy as sp

from .. import expressions, quality, sampling
from ..config import GenerationConfig
from ..sampling import DataGroup
from ..symbols import make_symbols
from ..utils import expr_text


def reference_meta(true_expr: sp.Expr, cfg: GenerationConfig,
                   var_symbols: list) -> dict:
    c = quality.complexity(true_expr)
    return {
        "expr": expr_text(true_expr, cfg.precision),
        "var_symbols": [str(s) for s in var_symbols],
        "skeleton": quality.skeleton_key(true_expr),
        "difficulty": quality.difficulty(true_expr),
        "ops": c["ops"],
        "depth": c["depth"],
        "vars": c["vars"],
    }


def sample_rows(true_expr: sp.Expr, cfg: GenerationConfig, var_symbols: list,
                seed: int) -> list[DataGroup]:
    """采样数据矩阵，带重试以过滤非有限值/无法求值的病态表达式。"""
    for attempt in range(20):
        try:
            return sampling.sample_data(
                true_expr, group=cfg.group, n_per_group=cfg.n_per_group,
                dim=cfg.dim, dist=cfg.dist, dom=cfg.dom, noise=cfg.noise,
                seed=seed + attempt, var_symbols=var_symbols,
                precision=cfg.precision,
            )
        except Exception:
            continue
    raise ValueError("采样的表达式数值不稳定，无法生成可用数据")


def make_var_symbols(cfg: GenerationConfig) -> list:
    return make_symbols(cfg.dim)


def data_payload(rows: list[DataGroup], cfg: GenerationConfig) -> dict:
    """把分组数据压成 JSON 友好的 data dict（供 GRPO reward 环境）。"""
    return {
        "groups": [
            {"variables": g.variables, "output": g.output, "dist": g.dist,
             "dom": g.dom, "noise": g.noise}
            for g in rows
        ],
        "dist": cfg.dist,
        "dom": cfg.dom,
        "dim": cfg.dim,
    }