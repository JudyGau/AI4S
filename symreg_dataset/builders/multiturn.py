"""综合多轮链（Multi-turn）数据构造器。

结构：problem(human) -> 分析+中间假设(gpt) -> 反馈(human) -> 最终表达式(gpt)。

- 首轮 gpt：给出定性分析 + 一个中间假设表达式 f1；
- 中间 human：反馈 f1 不够贴合并要求修正；
- 末轮 gpt：收敛到**正确**的参考表达式。
用于训练模型掌握「先分析假设、依反馈自我修正」的多步推理能力。

中间假设的生成经过两层质量约束，使"修正"这一轮具备真实语义、而非无意义或乱猜：
1. 结构近邻：用 `_perturb(keep_close=True)` 只产生与目标结构相近的改动
   （改单系数 / sin<->cos 换算子，必要时整体小幅缩放平移），排除 -expr 大幅翻转；
2. 数值质量带：在采样数据上用 `reward` 的数值项（num）打分，把候选限制在
   「明显劣于目标、但仍有相当解释力」的区间内：
   - 上限 `target - ε`：排除数值上约等于目标的候选（修正无意义）；
   - 下限 `0.25 * target`：排除完全乱猜（如一换算子就把函数毁掉的候选）；
   并在区间内取 **num 最高的候选**作为中间假设——它是最贴近目标的可信"首猜"。
   注意频域等系数改动是"大杠杆"，小幅扰动就会让 num 明显下降，故下限用 0.25×target 而非更严的阈值。
"""
from __future__ import annotations

import random

from .. import instructions, reward
from ..config import GenerationConfig
from ..schemas import make_multiturn_item, human, gpt
from ..utils import expr_text
from ._common import reference_meta, sample_rows, make_var_symbols
from .dpo import _perturb


def _num_quality(txt: str, true_expr, rows, var_symbols) -> float:
    """候选表达式在该样本数据上的数值奖励（R²/容差精度合成，0~1）。"""
    var_mat = rows[0].variables
    ref_ys = rows[0].output
    return reward.compute_reward(txt, true_expr, var_mat, ref_ys, var_symbols)["num"]


def build_multiturn_items(expressions, cfg: GenerationConfig,
                          seed: int = 0) -> list[dict]:
    rng = random.Random(cfg.seed + seed)
    var_symbols = make_var_symbols(cfg)
    items = []
    for i, true_expr in enumerate(expressions):
        rows = sample_rows(true_expr, cfg, var_symbols, seed=cfg.seed + i)
        problem = instructions.sft_instruction(rows, cfg.dim, rng, cfg.precision)
        final_txt = expr_text(true_expr, cfg.precision)

        target_num = _num_quality(final_txt, true_expr, rows, var_symbols)
        upper = target_num - max(0.015, 0.05 * target_num)  # ≈目标的候选不算假设
        floor = max(0.02, 0.25 * target_num)                # 低于此视为乱猜

        best = None                    # (num, txt)：区间内的最优近邻
        seen = set()
        for _ in range(100):
            cand = _perturb(true_expr, rng, keep_close=True)
            txt = expr_text(cand, cfg.precision)
            if txt == final_txt or txt in seen:
                continue
            seen.add(txt)
            cand_num = _num_quality(txt, true_expr, rows, var_symbols)
            if cand_num < floor or cand_num >= upper:
                continue
            if best is None or cand_num > best[0]:
                best = (cand_num, txt)

        if best is None:
            continue  # 连合理假设都取不到，跳过该样本（不满足训练质量的未生成）
        hypo_txt = best[1]

        conversations = [
            human(problem),
            gpt(instructions.analysis_and_hypothesis(hypo_txt, rng)),
            human(instructions.feedback_turn(rng)),
            gpt(final_txt),  # 收敛到正确表达式
        ]
        item = make_multiturn_item(
            conversations, reference=reference_meta(true_expr, cfg, var_symbols),
            system=cfg.system,
        )
        items.append(item)
    return items