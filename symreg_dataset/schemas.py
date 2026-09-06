"""对齐 LLaMA-Factory 的输出 schema（三种范式）。"""
from __future__ import annotations

import sympy as sp

from .utils import expr_text


def human(value: str) -> dict:
    return {"from": "human", "value": value}


def gpt(value: str) -> dict:
    return {"from": "gpt", "value": value}


def make_sft_item(instruction: str, target_expr: sp.Expr, reference: dict,
                  system: str = "") -> dict:
    """SFT：单轮 human->gpt。"""
    item = {
        "system": system,
        "conversations": [human(instruction), gpt(expr_text(target_expr))],
    }
    if reference:
        item["reference"] = reference  # 附加元数据（可选，不参与 SFT loss 过滤）
    return item


def make_dpo_item(instruction: str, chosen: sp.Expr, rejected: sp.Expr,
                  system: str = "") -> dict:
    """DPO：同指令下 chosen/rejected。"""
    return {
        "system": system,
        "instruction": instruction,
        "input": "",
        "chosen": [gpt(expr_text(chosen))],
        "rejected": [gpt(expr_text(rejected))],
    }


def make_grpo_item(problem: str, reference_expr: sp.Expr, data: dict,
                   system: str = "") -> dict:
    """GRPO：问题式（无标准答案的 human 提示）+ 供 reward 环境使用的 reference/data。"""
    return {
        "system": system,
        "conversations": [human(problem)],
        # 以下字段供自定义 reward 函数（scripts/integrate_reward.py）使用
        "reference": expr_text(reference_expr),
        "data": data,
    }