import re

from symreg_dataset.config import GenerationConfig
from symreg_dataset.builders import build_multiturn_items
from symreg_dataset.builders.dpo import _perturb
from symreg_dataset import expressions, reward


def _cfg(**kw):
    kw.setdefault("n_per_group", 16)
    kw.setdefault("seed", 0)
    return GenerationConfig(**kw)


def _extract_hypo(convs) -> str | None:
    m = re.search(r"f1 = ([^，,]+)", convs[1]["value"])
    return m.group(1).strip() if m else None


def test_multiturn_four_segments_and_converges():
    # 随机表达式可能因取不到合理假设被跳过（文档化的省略行为），故重试若干次取一个可出样本的
    items = None
    for _ in range(100):
        cand = expressions.random_expression(dim=1, max_depth=3)
        res = build_multiturn_items([cand], _cfg())
        if len(res) == 1:
            items = res
            break
    assert items is not None and len(items) == 1, "应能在重试内取到一个可出样本的表达式"
    conv = items[0]["conversations"]
    assert len(conv) == 4
    assert conv[0]["from"] == "human"
    assert "f1 = " in conv[1]["value"]
    assert conv[1]["from"] == "gpt"
    assert conv[2]["from"] == "human"
    assert conv[3]["from"] == "gpt"
    assert conv[3]["value"] == items[0]["reference"]["expr"]


def test_hypothesis_is_near_miss_quality_band():
    # 中间假设应"严格劣于目标（修正有意义）且具一定解释力（非乱猜）"
    e = None
    items = None
    cfg = _cfg()
    for _ in range(100):
        cand = expressions.random_expression(dim=1, max_depth=3)
        res = build_multiturn_items([cand], cfg)
        if len(res) == 1:
            e, items = cand, res
            break
    assert items is not None and len(items) == 1, "应能在重试内取到一个可出样本的表达式"
    conv = items[0]["conversations"]
    hypo = _extract_hypo(conv)
    assert hypo is not None and hypo != conv[3]["value"]

    # 重建 rows 以计算 num（与构造器同一采样种子可复现）
    from symreg_dataset.builders._common import sample_rows, make_var_symbols
    vs = make_var_symbols(cfg)
    rows = sample_rows(e, cfg, vs, seed=cfg.seed + 0)
    tgt = reward.compute_reward(conv[3]["value"], e, rows[0].variables,
                                rows[0].output, vs)["num"]
    hy = reward.compute_reward(hypo, e, rows[0].variables,
                               rows[0].output, vs)["num"]
    assert hy < tgt - max(0.015, 0.05 * tgt), "中间假设应严格劣于目标"
    assert hy >= max(0.02, 0.25 * tgt), "中间假设不应是完全乱猜"


def test_perturb_keep_close_excludes_full_flip():
    # keep_close 模式应产出与目标结构近邻的候选，而不是整体 -expr 翻转
    e = expressions.random_expression(dim=1, max_depth=4)
    import random
    rng = random.Random(0)
    for _ in range(50):
        cand = _perturb(e, rng, keep_close=True)
        assert not cand.equals(-e), f"keep_close 不应产生 -expr 翻转: {cand}"