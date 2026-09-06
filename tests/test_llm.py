import random

import sympy as sp

from symreg_dataset import llm, instructions
from symreg_dataset.config import GenerationConfig
from symreg_dataset.symbols import make_symbols
from symreg_dataset.builders._common import sample_rows, make_var_symbols
from symreg_dataset.utils import expr_text
from symreg_dataset.builders.dpo import _perturb
from symreg_dataset.providers import LLMConfigError


class _FakeProvider:
    """按调用顺序返回预先编排的回复，模拟 LLM 聊天。"""
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.i = 0

    def chat(self, messages, temperature=0.7, max_tokens=256) -> str:
        if self.i >= len(self.replies):
            raise LLMConfigError("no more replies")
        r = self.replies[self.i]
        self.i += 1
        return r


def _cfg(**kw):
    kw.setdefault("n_per_group", 16)
    kw.setdefault("seed", 0)
    return GenerationConfig(**kw)


def _target_expr():
    x = make_symbols(1)[0]
    return sp.cos(sp.Float(2.7) * x) + sp.sin(sp.Float(0.5) * x)


def test_extract_expr_f_equals_and_whole():
    assert llm._extract_expr("分析：周期性。\nf = sin(x)**2", 2) == \
        llm._canonical("sin(x)**2", 2)
    # 整体可解析（无 f= 前缀）
    assert llm._extract_expr("cos(x) + 1", 2) == llm._canonical("cos(x) + 1", 2)


def test_build_provider_missing_config():
    cfg = _cfg()
    try:
        llm.build_provider(cfg)
        assert False, "应当因配置缺失抛 LLMConfigError"
    except LLMConfigError:
        pass


def test_generate_chain_converges_after_refinement():
    e = _target_expr()
    cfg = _cfg(use_llm=True, llm_base_url="http://x", llm_api_key="k",
               llm_model="m", llm_max_rounds=4)
    vs = make_var_symbols(cfg)
    rows = sample_rows(e, cfg, vs, seed=cfg.seed)
    final = expr_text(e, cfg.precision)
    rng = random.Random(0)
    hypo = None
    for _ in range(50):
        h = expr_text(_perturb(e, rng, keep_close=True), cfg.precision)
        if h != final:
            hypo = h
            break
    assert hypo is not None
    fake = _FakeProvider([f"初步判断为周期函数。\nf = {hypo}",
                          f"修正后收敛。\nf = {final}"])
    problem = instructions.sft_instruction(rows, cfg.dim, rng, cfg.precision)
    conv = llm.generate_chain(problem, e, rows, fake, cfg, final, vs)

    assert conv is not None
    # 长度 >= 4：problem + 假设 + 反馈 + 收敛
    assert len(conv) >= 4
    assert conv[-1]["from"] == "gpt"
    assert llm._extract_expr(conv[-1]["value"], cfg.precision) == final
    # 中间含一条 human 反馈
    assert any(t["from"] == "human" for t in conv[1:-1])


def test_generate_chain_returns_none_when_first_is_correct():
    e = _target_expr()
    cfg = _cfg(use_llm=True, llm_base_url="x", llm_api_key="k",
               llm_model="m", llm_max_rounds=4)
    vs = make_var_symbols(cfg)
    rows = sample_rows(e, cfg, vs, seed=cfg.seed)
    final = expr_text(e, cfg.precision)
    fake = _FakeProvider([f"首轮即给出。\nf = {final}"])
    problem = "数据占位"
    assert llm.generate_chain(problem, e, rows, fake, cfg, final, vs) is None


def test_relaxed_chain_accepts_improving_best():
    # 首猜质量低、后续某轮明显更好且达到下限 → 接受，末轮为最佳假设
    cfg = _cfg(llm_relaxed=True, llm_relaxed_floor=0.4, llm_relaxed_gain=0.05)
    target = 0.99
    rounds = [
        ("分析f = a*x**2+b*x+c", "a*x**2", 0.1),
        ("改为f = sin(x)+cos(x)", "sin(x)+cos(x)", 0.62),
        ("再修f = 2*sin(x)+2*cos(x)", "2*sin(x)+2*cos(x)", 0.58),
    ]
    conv = llm._relaxed_chain(rounds, "部分点位待拟合 {x=[1,2]}", cfg, target)
    assert conv is not None
    assert conv[0]["from"] == "human"
    # 末轮 gpt = 最佳假设那轮（0.62）
    assert "sin(x)+cos(x)" in conv[-1]["value"]


def test_relaxed_chain_rejects_no_improvement():
    cfg = _cfg(llm_relaxed=True, llm_relaxed_floor=0.4, llm_relaxed_gain=0.05)
    # 最佳未比首猜提升 ≥ gain → 拒绝
    rounds = [("f = a*x", "a*x", 0.3), ("f = b", "b", 0.32)]
    assert llm._relaxed_chain(rounds, "p {x=[1,2]}", cfg, 0.99) is None
    # 达到下限但增量不足
    rounds2 = [("f = x", "x", 0.7), ("f = 1.5*x", "1.5*x", 0.73)]
    assert llm._relaxed_chain(rounds2, "p {x=[1,2]}", cfg, 0.99) is None


def test_relaxed_chain_flags_below_floor_and_nan():
    cfg = _cfg(llm_relaxed=True, llm_relaxed_floor=0.6, llm_relaxed_gain=0.05)
    # 提升够但低于下限
    rounds = [("f = x", "x", 0.1), ("f = 1.5*x", "1.5*x", 0.5)]
    assert llm._relaxed_chain(rounds, "p {x=[1,2]}", cfg, 0.99) is None
    # 全部 NaN → 拒绝
    nan_rounds = [("f = x", "x", float("nan")), ("f = y", "y", 0.5)]
    assert llm._relaxed_chain(nan_rounds, "p {x=[1,2]}", cfg, 0.99) is None


def test_build_llm_fallback_to_template(monkeypatch):
    # LLM 网络/配置失败 → build_multiturn_items 回退到模板路径仍能出样本
    e = _target_expr()
    cfg = _cfg(use_llm=True, llm_base_url="x", llm_api_key="k",
               llm_model="m", llm_max_rounds=2, llm_fallback=True)

    class _BadProvider:
        def chat(self, messages, temperature=0.7, max_tokens=256) -> str:
            raise LLMConfigError("network down")

    import symreg_dataset.llm as llm_mod
    from symreg_dataset.builders.multiturn import build_multiturn_items
    monkeypatch.setattr(llm_mod, "build_provider", lambda c: _BadProvider())

    items = build_multiturn_items([e], cfg, seed=0)
    assert len(items) >= 1
    conv = items[0]["conversations"]
    assert conv[-1]["from"] == "gpt"
    assert llm._extract_expr(conv[-1]["value"], cfg.precision) == \
        expr_text(e, cfg.precision)