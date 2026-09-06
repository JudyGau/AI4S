"""GRPO reward 接入示例（可用于 LLaMA-Factory 自定义 reward 函数或独立环境）。

`compute_reward` 对模型输出列表逐条计算：解析表达式 -> 在数据上做数值拟合 + 形式一致性。
字段与 builders/grpo.py 写出的 JSONL 结构对应。
"""
from __future__ import annotations

from symreg_dataset import reward
from symreg_dataset.normalize import parse_expression_text
from symreg_dataset.symbols import make_symbols


def make_var_symbols(dim: int):
    return make_symbols(dim)


def compute_reward(model_outputs, task_labels, w_num=0.6, w_form=0.4, tau=0.05):
    """LLaMA-Factory reward 风格接口。

    - model_outputs: 每项为字符串，是模型对一个 problem 的输出序列（取末尾表达式行）。
    - task_labels: 与 JSONL 中 `reference` / `data` 对应，需在加载数据时挂到 batch 上。
    """
    rewards = []
    for i, out in enumerate(model_outputs):
        label = task_labels[i] if i < len(task_labels) else None
        if label is None:
            rewards.append(0.0)
            continue
        ref_text = label["reference"]           # 目标表达式字符串
        groups = label["data"]["groups"]        # 数据矩阵
        dim = label["data"]["dim"]
        var_symbols = make_var_symbols(dim)
        g0 = groups[0]
        var_mat, ref_ys = g0["variables"], g0["output"]
        ref_expr = parse_expression_text(ref_text)
        # 从多行输出中摘取最后一行作为候选表达式
        cand = out.strip().splitlines()[-1] if out.strip() else ""
        r = reward.compute_reward(cand, ref_expr, var_mat, ref_ys,
                                  var_symbols, w_num, w_form, tau)
        rewards.append(r["total"])
    return rewards


if __name__ == "__main__":
    # 自测：chosen 表达式应获得高奖励，随机乱猜应获得低奖励
    outs = ["f = sin(x) + cos(x)", "f = x"]
    labels = [{
        "reference": "sin(x) + cos(x)",
        "data": {"dim": 1, "groups": [{"variables": [[0.5]], "output": [1.357]}]},
    }] * 2
    print(compute_reward(outs, labels))