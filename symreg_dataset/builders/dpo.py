"""DPO 数据构造器：同一指令下 chosen（正确）/ rejected（扰动/负样本）。

rejected 通过对正确表达式做扰动生成（改系数、换算子等），确保它仍是"看似合理但错误"的负样本；
并用 reward.compute_reward 校验 chosen 奖励严格优于 rejected，剔除不合格对。
"""
from __future__ import annotations

import random

import sympy as sp

from .. import instructions, reward
from ..config import GenerationConfig
from ..schemas import make_dpo_item
from ..utils import expr_text
from ._common import sample_rows, make_var_symbols


def _outs(expr, cfg: GenerationConfig) -> str:
    """与写盘一致的序列化字符串。"""
    return expr_text(expr, cfg.precision)


def _perturb(expr: sp.Expr, rng: random.Random, keep_close: bool = False) -> sp.Expr:
    """生成扰动负样本：随机改系数，或对子表达式换算子/换符号。

    keep_close=True：仅保留结构近邻的改动（改系数/换算子，必要时整体小幅缩放/平移），
    用于多轮链的"中间假设"（看似合理但不完美）；否则额外加入整体 -expr 翻转这类硬负样本（供 DPO）。
    """
    candidates: list[sp.Expr] = []

    # 1) 改系数：给某个数值常数加一个小扰动（近邻改动）
    numbers = sorted((n for n in expr.atoms(sp.Number) if n not in (sp.Integer(0), sp.Integer(1))),
                     key=lambda n: str(float(n)))
    if numbers:
        val = numbers[rng.randrange(len(numbers))]
        delta = round(rng.uniform(0.15, 0.6), 2) * rng.choice([-1, 1])
        try:
            new_val = sp.Float(round(float(val) + delta, 2))
            candidates.append(expr.xreplace({val: new_val}))
        except Exception:
            pass

    # 2) 换一元算子：sin <-> cos，exp <-> sin 等（近邻改动）
    replacements = [("sin", "cos"), ("cos", "sin"), ("exp", "sin"), ("log", "cos")]
    s = str(expr)
    for a, b in replacements:
        if a in s and b not in s:
            candidates.append(sp.sympify(s.replace(a, b, 1)))
            break

    if not candidates:
        if keep_close:
            # 既无系数可改也无算子可换：整体小幅缩放/平移，仍属近邻
            try:
                if rng.random() < 0.5:
                    return sp.Float(round(rng.uniform(0.5, 1.5), 2)) * expr
                return expr + sp.Float(rng.choice([-0.3, 0.3, 0.7, -0.7]))
            except Exception:
                return -expr  # 极端兜底
        # 3) 符号翻转：整体乘 -1（硬负样本，仅非近邻模式）
        candidates.append(-expr)

    return rng.choice(candidates)


def build_dpo_items(expressions, cfg: GenerationConfig,
                    seed: int = 0) -> list[dict]:
    rng = random.Random(cfg.seed + seed)
    var_symbols = make_var_symbols(cfg)
    items = []
    for i, true_expr in enumerate(expressions):
        rows = sample_rows(true_expr, cfg, var_symbols, seed=cfg.seed + i)
        instruction = instructions.sft_instruction(rows, cfg.dim, rng, cfg.precision)
        var_mat = rows[0].variables
        ref_ys = rows[0].output
        chosen_txt = sp.sstr(true_expr)

        rejected_expr = None
        for _ in range(20):
            cand = _perturb(true_expr, rng)
            # 与写盘使用的序列化一致（舍入到 cfg.precision）判重，保证 chosen/rejected 真正不同
            if _outs(cand, cfg) == _outs(true_expr, cfg):
                continue
            r_chosen = reward.compute_reward(chosen_txt, true_expr, var_mat, ref_ys, var_symbols)
            r_rej = reward.compute_reward(sp.sstr(cand), true_expr, var_mat, ref_ys, var_symbols)
            # chosen 必须严格优于 rejected（数值更差或形式更差）
            if r_chosen["total"] > r_rej["total"] or r_rej["rmse"] > r_chosen["rmse"]:
                rejected_expr = cand
                break
        if rejected_expr is None:
            continue
        items.append(make_dpo_item(instruction, true_expr, rejected_expr, system=cfg.system))
    return items