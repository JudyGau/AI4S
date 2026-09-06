"""指令模板（多套措辞，增强数据多样性）。

覆盖 SFT / DPO / GRPO 三种范式。点位按固定精度格式化以保证可复现。
"""
from __future__ import annotations

import random
from typing import Sequence

from .sampling import DataGroup


def format_points(groups: Sequence[DataGroup], dim: int, precision: int = 2) -> str:
    """把数据矩阵格式化为 LLM 可读文本（可含多组）。
    形如：x1 = [...], x2 = [...] 以及 y = [...]。
    """
    blocks = []
    for g in groups:
        var_parts = []
        for d in range(dim):
            var_parts.append(f"x{d + 1} = {g.variables[d]}")
        blocks.append(f"{', '.join(var_parts)}, y = {g.output}")
    if len(blocks) == 1:
        return blocks[0]
    return "；".join(f"第{i + 1}组: {b}" for i, b in enumerate(blocks))


SFT_TEMPLATES = [
    "请对以下数据进行符号回归：给定自变量 {vars} 与因变量 y 的观测值 {points}，"
    "请给出一个简洁且能拟合这些数据的数学表达式 f。仅输出 f = 最简表达式。",
    "进行符号回归：{points}。请基于这些数据点，给出自变量与因变量之间的函数关系，"
    "结果以最简的数学表达式形式输出。",
    "给定数据点：{points}。请推断出它们背后满足的数学表达式 f({vars_arg}) = ...，"
    "要求尽量简洁、符合真实科学规律，只输出表达式本身。",
]

GRPO_TEMPLATES = [
    "请对给定数据点 {points} 做符号回归，推导出自变量与因变量之间的数学表达式。"
    "你只需以 f = 表达式 的形式输出最符合数据的表达式，不要解释。",
    "数据：{points}。请你给出一个能描述它们满足的函数关系 f = ...，"
    "优先选择更简单、更符合物理直觉的表达式。",
]

DPO_TEMPLATES = SFT_TEMPLATES  # DPO 复用 SFT 指令

# ---------------------------- 多轮对话 ----------------------------
# 综合分析链：首轮 分析+中间假设 -> 反馈 -> 收敛到最终表达式
REASON_PHRASES = [
    "明显的周期/振荡（含三角函数）",
    "幂函数或多项式趋势",
    "指数增长/衰减",
    "含三角函数与幂的组合",
    "带对数/平方根的非线性",
]

ANALYSIS_TEMPLATES = [
    "我先对数据做初步分析：整体更像是{reason}。基于此，先给出一个候选表达式 f1 = {hypo}，作为第一轮假设。",
    "初步观察，数据大致呈{reason}特征。我先抛出一个中间候选 f1 = {hypo}，供后续修正。",
    "就数据形态而言更接近{reason}。我先生成一个试探性表达式 f1 = {hypo}，后续可进一步校正。",
]

FEEDBACK_TEMPLATES = [
    "候选 f1 在若干数据点上拟合仍偏离，未能充分捕捉整体趋势。请在综合全部数据点后，给出更准确、更简洁的**最终**表达式 f = ...",
    "初步假设还不够贴合数据，尤其在中段取值范围误差明显。请直接给出收敛后的**最终**表达式 f = ...",
    "这个 f1 离目标函数还有差距。请修正系数和算子，输出最符合数据规律的**最终**表达式 f = ...",
]


def _var_names(dim: int) -> str:
    return "x" if dim == 1 else ", ".join(f"x{i + 1}" for i in range(dim))


def _var_arg(dim: int) -> str:
    return "x" if dim == 1 else ", ".join(f"x{i + 1}" for i in range(dim))


def make_instruction(templates: list[str], rows: Sequence[DataGroup], dim: int,
                     rng: random.Random, precision: int = 2) -> str:
    points = format_points(rows, dim, precision)
    tmpl = rng.choice(templates)
    return tmpl.format(points=points, vars=_var_names(dim), vars_arg=_var_arg(dim))


def sft_instruction(rows: Sequence[DataGroup], dim: int,
                    rng: random.Random, precision: int = 2) -> str:
    return make_instruction(SFT_TEMPLATES, rows, dim, rng, precision)


def grpo_instruction(rows: Sequence[DataGroup], dim: int,
                     rng: random.Random, precision: int = 2) -> str:
    return make_instruction(GRPO_TEMPLATES, rows, dim, rng, precision)


def analysis_and_hypothesis(hypo_txt: str, rng: random.Random) -> str:
    """综合多轮链的第 1 轮回复：定性分析 + 中间假设 f1。"""
    reason = rng.choice(REASON_PHRASES)
    tmpl = rng.choice(ANALYSIS_TEMPLATES)
    return tmpl.format(reason=reason, hypo=hypo_txt)


def feedback_turn(rng: random.Random) -> str:
    """综合多轮链的第 2 轮人机反馈：督促修正到最终表达式。"""
    return rng.choice(FEEDBACK_TEMPLATES)