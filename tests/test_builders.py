from symreg_dataset.config import GenerationConfig
from symreg_dataset.builders import build_sft_items, build_grpo_items, build_dpo_items
from symreg_dataset import expressions, quality


def _make_cfg(**kw):
    kw.setdefault("noise", 0.0)
    kw.setdefault("group", 1)
    kw.setdefault("n_per_group", 16)
    kw.setdefault("seed", 0)
    return GenerationConfig(**kw)


def test_sft_items_schema():
    e = expressions.random_expression(dim=1, max_depth=3)
    cfg = _make_cfg()
    items = build_sft_items([e], cfg)
    assert len(items) == 1
    it = items[0]
    assert it["conversations"][0]["from"] == "human"
    assert it["conversations"][1]["from"] == "gpt"
    assert "reference" in it


def test_grpo_items_schema():
    e = expressions.random_expression(dim=1, max_depth=3)
    cfg = _make_cfg()
    items = build_grpo_items([e], cfg)
    it = items[0]
    assert it["conversations"][0]["from"] == "human"      # 问题式，无 gpt 答案
    assert "reference" in it
    assert "data" in it


def test_dpo_chosen_better():
    cfg = _make_cfg()
    items = build_dpo_items([expressions.random_expression(dim=1, max_depth=3)], cfg)
    assert len(items) >= 1
    it = items[0]
    assert it["chosen"] != it["rejected"]