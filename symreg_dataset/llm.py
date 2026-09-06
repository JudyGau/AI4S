"""LLM 多轮对话生成：OpenAI 兼容 API + 奖励门控收敛。

真对话产出「分析 -> 假设 -> 依反馈修正 -> 收敛」的多轮轨迹；
`compute_reward` 在**真实数据**上判定每轮质量并生成数值反馈；只有最终轮收敛到目标
表达式的链才会被保留。API 调用失败、输出不合规或未收敛时，可由调用方经 `fallback`
回退到模板路径，保证离线可跑。

不新增任何 schema——复用 `make_multiturn_item` 与现有 conversations 结构。
"""
from __future__ import annotations

import random

from . import instructions, reward
from .config import GenerationConfig
from .schemas import make_multiturn_item, human, gpt
from .utils import expr_text
from .normalize import parse_expression_text
from .builders._common import reference_meta, sample_rows, make_var_symbols
from .providers import LlmProvider, OpenAiCompatible, LLMConfigError, LLMError

SYSTEM_PROMPT = (
    "你是擅长符号回归的科学助手。重要约定：你需要给出数学表达式，且回答的"
    "最后一行必须写成：f = 最简表达式。不要解释最终行以外的格式。"
)

ROUND1_PROMPT = (
    "请对以下数据做符号回归：{points}\n"
    "先简短分析这组数据的形态（函数族 / 趋势），并给出你的初步假设表达式，"
    "且回答的最后一行写成：f = 表达式。"
)


def build_provider(cfg: GenerationConfig) -> LlmProvider:
    if not (cfg.llm_base_url and cfg.llm_api_key and cfg.llm_model):
        raise LLMConfigError("需要配置 llm_base_url / llm_api_key / llm_model")
    return OpenAiCompatible(cfg.llm_base_url, cfg.llm_api_key,
                            cfg.llm_model, cfg.llm_timeout,
                            extra_body=cfg.llm_extra_body)


def _canonical(txt: str, precision: int) -> str | None:
    """把文本解析成规范化表达式字符串；失败返回 None。"""
    try:
        return expr_text(parse_expression_text(txt), precision)
    except Exception:
        return None


def _normalize_output(text: str) -> str:
    """把模型输出规整成可解析的候选文本：
    去反引号/井号记号、Unicode 上下标、数学符号、花括号等式写法等（**保留 `*` 作为幂）。"""
    t = text.replace("`", "").replace("#", "").strip()
    # Unicode 上标/下标（x²、x³、x⅕ 等）→ ** 幂
    sup = {"²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
           "⁷": "7", "⁸": "8", "⁹": "9", "⁰": "0"}
    for u, d in sup.items():
        t = t.replace(u, f"**{d}")
    # 数学符号
    t = t.replace("×", "*").replace("·", "*").replace("−", "-")
    t = t.replace("（", "(").replace("）", ")").replace("＝", "=")
    t = t.replace("^", "**")  # 非思考模型常用 x1^2 这种 ASCII 幂写法
    # 容忍 f(x) = 这类前缀（解析器已处理，这里再兜底去成对花括号）
    t = t.replace("{", "").replace("}", "")
    return t


def _strip_markdown(expr: str) -> str:
    """去除候选表达式首尾的 markdown ** 强调记号（而不影响内部的幂 **）。"""
    return expr.strip().strip("*").strip()


def _extract_expr(reply: str, precision: int) -> str | None:
    """从模型回复中提取表达式。

    优先取最后一行 / 最后一次出现的 `f = ...`（含 GLM 的 `f(x) = ...` 或藏在 markdown 里），
    否则尝试整体解析。
    """
    if not reply:
        return None
    candidates = []
    for line in reply.splitlines():
        cand = _parse_candidate(line, precision)
        if cand is not None:
            candidates.append(cand)
    if candidates:
        return candidates[-1]
    # 行内也存在 `f = expr`（未换行）：取最后一段
    import re as _re
    m = list(_re.finditer(r"f(?:\([^)]*\))?\s*=\s*([^\n]+)", reply))
    if m:
        return _canonical(_strip_markdown(_normalize_output(m[-1].group(1))), precision)
    return _canonical(_strip_markdown(_normalize_output(reply.strip())), precision)


def _parse_candidate(line: str, precision: int) -> str | None:
    """若该行形如 `f = expr` / `f(x) = expr`（容忍 `**` 等前缀与 Unicode），返回规范化串。"""
    t = _normalize_output(line).strip()
    low = t.lstrip("*")
    for prefix in ("f=", "f =", "f(x)=", "f(x) ="):
        if low.startswith(prefix):
            cand = _strip_markdown(t[t.find("=") + 1:].strip())
            if cand:
                return _canonical(cand, precision)
    return None


def _num_quality(txt: str, true_expr, rows, var_symbols) -> float:
    var_mat = rows[0].variables
    ref_ys = rows[0].output
    return reward.compute_reward(txt, true_expr, var_mat, ref_ys, var_symbols)["num"]


def make_feedback(num: float) -> str:
    """基于真实数值质量生成反馈；只回指标，不回答案，避免修正退化成"照着念"。"""
    if num < 0.3:
        return (
            f"你给出的候选在数据上几乎无拟合（数值质量≈{num:.2f}），函数族可能判断"
            f"有误，请重新判断。回答的最后一行仍写成：f = 表达式。"
        )
    if num < 0.8:
        return (
            f"候选结构接近，但数值仍偏差（数值质量≈{num:.2f}）。请修正系数与算子，"
            f"回答的最后一行仍写成：f = 表达式。"
        )
    return (
        f"非常接近了（数值质量≈{num:.2f}），请做最终精确修正并收敛。"
        f"回答的最后一行仍写成：f = 表达式。"
    )


def _converged(num: float, target_num: float, tol: float = 1e-3) -> bool:
    """数值收敛判定：候选在数据上的拟合质量足够接近参考表达式的自身质量。

    自然语言模型常把 `2.0*x` 写成 `2*x`、把变量写成 `x1` 等，字符串不相等但数值等价，
    因此用数值门槛代替字符串相等判定。
    """
    if num != num:  # NaN
        return False
    margin = max(tol, tol * max(target_num, 1e-6))
    return num >= target_num - margin


def _relaxed_chain(rounds: list[tuple[str, str, float]], problem: str,
                   cfg: GenerationConfig, target_num: float) -> list[dict] | None:
    """放宽收敛：取循环内"最佳拟合"假设作为末轮答案，产出真实多轮改进链。

    rounds: [(reply, expr_c, num)]，仅含能解析出表达式的轮次。
    要求最佳假设在数值上明显优于首轮假设（体现"依反馈修正"），且非垃圾拟合；
    否则返回 None（无价值的链交给回退模板）。"""
    finite = [(i, n) for i, n in enumerate(rounds) if n[2] == n[2]]  # 排除 NaN
    if len(finite) < 2:
        return None
    best_i = max(finite, key=lambda p: p[1][2])[0]
    best_num = rounds[best_i][2]
    first_num = rounds[0][2]
    margin = max(1e-3, 1e-3 * max(target_num, 1e-6))
    too_close = best_num >= target_num - margin   # 已数值收敛，本不该走到这
    improved = best_num >= first_num + cfg.llm_relaxed_gain
    good_enough = best_num >= cfg.llm_relaxed_floor
    if not improved or not good_enough or too_close:
        return None
    turns = [human(problem)]
    for k in range(best_i):
        turns.append(gpt(rounds[k][0]))
        turns.append(human(make_feedback(rounds[k][2])))
    turns.append(gpt(rounds[best_i][0]))
    return turns


def _alias_vars(text: str, var_symbols) -> str:
    """把模型常用的数值后缀变量 `x1, x2, ...` 规整为数据列对应的参考符号名。"""
    import re
    names = [str(s) for s in var_symbols]

    def repl(m):
        i = int(m.group(1))
        return names[i - 1] if 1 <= i <= len(names) else m.group(0)

    return re.sub(r"\bx(\d+)\b", repl, text)


def generate_chain(problem: str, true_expr, rows, provider: LlmProvider,
                   cfg: GenerationConfig, final_txt: str,
                   var_symbols) -> list[dict] | None:
    """奖励门控循环。

    返回（收敛的）conversations 列表（末段为正确表达式）；首轮即正确、未收敛或异常返回 None。
    要求首轮为一个"不完美假设"，否则不成多轮链。收敛以数值拟合质量为准（见 `_converged`）。
    """
    target_num = _num_quality(final_txt, true_expr, rows, var_symbols)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ROUND1_PROMPT.format(points=problem)},
    ]
    turns: list[dict] = [human(problem)]
    rounds: list[tuple[str, str, float]] = []   # 能解析出表达式的轮次 (reply, expr_c, num)
    first_round = True
    for _ in range(max(1, cfg.llm_max_rounds)):
        try:
            reply = provider.chat(msgs, cfg.llm_temperature,
                                  max_tokens=cfg.llm_max_tokens)
        except LLMError:
            return _relaxed_chain(rounds, problem, cfg, target_num) \
                if cfg.llm_relaxed else None

        expr_c = _extract_expr(reply, cfg.precision)
        msgs.append({"role": "assistant", "content": reply})
        turns.append(gpt(reply))

        if expr_c is None:
            msgs.append({"role": "user",
                         "content": "未能解析出表达式。回答的最后一行必须写成：f = 表达式。请重试。"})
            continue

        expr_c = _alias_vars(expr_c, var_symbols)
        try:
            num = _num_quality(expr_c, true_expr, rows, var_symbols)
        except Exception:
            num = float("nan")   # 如含未赋值符号系数的表达式求值失败：该轮视为无效，但不断链
        rounds.append((reply, expr_c, num))

        if first_round:
            first_round = False
            if _converged(num, target_num):
                # 首轮即已数值收敛：没有"不完美假设"，不适合做多轮链
                return None
            fb = make_feedback(num)
            turns.append(human(fb))
            msgs.append({"role": "user", "content": fb})
            continue

        # 后续轮：数值收敛即成功
        if _converged(num, target_num):
            return turns
        fb = make_feedback(num)
        turns.append(human(fb))
        msgs.append({"role": "user", "content": fb})

    # 循环耗尽仍未精确收敛：放宽模式下接受"最佳改进拟合"
    return _relaxed_chain(rounds, problem, cfg, target_num) if cfg.llm_relaxed else None


def _build_one(provider, i, true_expr, cfg, seed, var_symbols, fallback):
    """构造单个目标的多轮样本；失败/无合适链返回 None。线程安全：用独立 rng。"""
    rng = random.Random(cfg.seed + seed + i)
    rows = sample_rows(true_expr, cfg, var_symbols, seed=cfg.seed + i)
    problem = instructions.sft_instruction(rows, cfg.dim, rng, cfg.precision)
    final_txt = expr_text(true_expr, cfg.precision)
    try:
        conv = generate_chain(problem, true_expr, rows, provider,
                              cfg, final_txt, var_symbols)
    except Exception:
        conv = None
    if conv is None and fallback is not None:
        conv = fallback(problem, true_expr, rows, rng,
                        var_symbols, final_txt, cfg)
    if conv is None:
        return None
    return make_multiturn_item(
        conv, reference=reference_meta(true_expr, cfg, var_symbols),
        system=cfg.system,
    )


def build_multiturn_llm_items(expressions, cfg: GenerationConfig,
                              seed: int = 0, fallback=None,
                              provider=None) -> list[dict]:
    """LLM 多轮生成入口；`fallback(problem,true_expr,rows,rng,var_symbols,final_txt,cfg)`
    在链生成失败时提供回退 conversations（如模板路径）。`cfg.llm_max_workers>1` 时用线程池并发。"""
    if provider is None:
        provider = build_provider(cfg)
    var_symbols = make_var_symbols(cfg)
    work = [(provider, i, e, cfg, seed, var_symbols, fallback)
            for i, e in enumerate(expressions)]
    if cfg.llm_max_workers > 1 and len(work) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=cfg.llm_max_workers) as ex:
            results = list(ex.map(lambda a: _build_one(*a), work))
    else:
        results = [_build_one(*a) for a in work]
    return [it for it in results if it is not None]