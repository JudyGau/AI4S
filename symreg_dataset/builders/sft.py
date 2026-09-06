"""SFT 数据构造器：单轮指令 -> 目标表达式。"""
from __future__ import annotations

import random

from .. import instructions
from ..config import GenerationConfig
from ..schemas import make_sft_item
from ._common import reference_meta, sample_rows, make_var_symbols


def build_sft_items(expressions, cfg: GenerationConfig,
                    seed: int = 0) -> list[dict]:
    rng = random.Random(cfg.seed + seed)
    var_symbols = make_var_symbols(cfg)
    items = []
    for i, true_expr in enumerate(expressions):
        rows = sample_rows(true_expr, cfg, var_symbols, seed=cfg.seed + i)
        instruction = instructions.sft_instruction(rows, cfg.dim, rng, cfg.precision)
        item = make_sft_item(
            instruction, true_expr,
            reference=reference_meta(true_expr, cfg, var_symbols),
            system=cfg.system,
        )
        items.append(item)
    return items