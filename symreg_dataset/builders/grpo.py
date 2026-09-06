"""Form-GRPO 数据构造器。

对齐 SymbArena / Symbolic-R1：问题式 human 提示（无标准答案），
附带 reference（目标表达式）与 data（数据矩阵），供自定义 reward 环境用 reward.py 打分，
奖励 = 数值项 + 形式项（Form-GRPO）。
"""
from __future__ import annotations

import random

from .. import instructions
from ..config import GenerationConfig
from ..schemas import make_grpo_item
from ._common import data_payload, sample_rows, make_var_symbols


def build_grpo_items(expressions, cfg: GenerationConfig,
                     seed: int = 0) -> list[dict]:
    rng = random.Random(cfg.seed + seed)
    var_symbols = make_var_symbols(cfg)
    items = []
    for i, true_expr in enumerate(expressions):
        rows = sample_rows(true_expr, cfg, var_symbols, seed=cfg.seed + i)
        problem = instructions.grpo_instruction(rows, cfg.dim, rng, cfg.precision)
        item = make_grpo_item(
            problem, true_expr,
            data=data_payload(rows, cfg),
            system=cfg.system,
        )
        items.append(item)
    return items