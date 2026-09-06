"""CLI 总入口：一次生成 SFT / GRPO / DPO 三种范式的 JSONL 数据集。

用法示例：
    python -m scripts.generate --n 20 --dim 1
    python -m scripts.generate --sft --dpo --n 100 --dim 2 --noise 0.05 --out data
"""
from __future__ import annotations

import argparse
import os
import random


def _parse_extra_body(text: str) -> dict | None:
    """解析 `--llm-extra-body` / LLM_EXTRA_BODY 的 JSON；空、非法返回 None。

    例：{"thinking":{"type":"enabled","budget_tokens":256}}
    """
    if not text:
        return None
    try:
        import json
        val = json.loads(text)
        return val if isinstance(val, dict) else None
    except ValueError:
        import warnings
        warnings.warn("LLM_EXTRA_BODY 不是合法 JSON 对象，已忽略")
        return None


import sympy as sp

from symreg_dataset.config import GenerationConfig
from symreg_dataset import expressions, quality, sampling, real_lib
from symreg_dataset.builders import (
    build_sft_items, build_grpo_items, build_dpo_items, build_multiturn_items,
)
from symreg_dataset.writers import save_jsonl
from symreg_dataset.utils import expr_text



def produce_expressions(cfg: GenerationConfig, rng: random.Random) -> list:
    """生成并过滤表达式：复杂度、骨架唯一性、采样稳定性、（可选）现实性。"""
    library = real_lib.load_real_equations() if cfg.use_real_lib else []
    var_symbols = None
    exprs: list = []
    attempts = 0
    max_attempts = max(50, cfg.n * 40)
    while len(exprs) < cfg.n and attempts < max_attempts:
        attempts += 1
        e = expressions.random_expression(cfg.dim, cfg.max_depth, rng=rng)
        # 0) 排除含符号无穷/NaN 的病态节点（会导致 numpy lambdify 打印失败）
        if e.has(sp.zoo, sp.nan, sp.oo, sp.S.NegativeInfinity):
            continue
        # 1) 复杂度过滤（树深）
        if not quality.is_complexity_ok(e, cfg.min_depth, cfg.max_depth_limit):
            continue
        # 2) 骨架唯一性
        if not quality.is_unique(e, seen):
            continue
        # 3) 采样稳定性 + 数值量级（非有限值或量级超大视为病态，跳过）
        try:
            rows = sampling.sample_data(e, group=1, n_per_group=cfg.n_per_group,
                                        dim=cfg.dim, dist=cfg.dist, dom=cfg.dom,
                                        noise=cfg.noise, seed=attempts,
                                        precision=cfg.precision)
        except ValueError:
            continue
        if any(abs(v) > cfg.max_abs for row in rows for v in row.output):
            continue
        # 4) 现实性软过滤
        if cfg.use_real_lib and not real_lib.is_realistic(e, cfg.real_lib_threshold, library):
            continue
        if quality.difficulty(e) == "easy" and cfg.sft:  # easy 数量太多时可精简，此处保留
            pass
        exprs.append(e)
    if len(exprs) < cfg.n:
        raise RuntimeError(f"只生成 {len(exprs)}/{cfg.n} 个通过质量过滤的表达式，请放宽参数")
    return exprs


seen: set[str] = set()


def split_sets(exprs, cfg: GenerationConfig, rng: random.Random):
    train, test = quality.split_by_skeleton(exprs, cfg.test_ratio, seed=cfg.seed)
    return train, test


def build_all(train, test, cfg: GenerationConfig):
    out = {}
    if cfg.sft:
        out["sft_train"] = build_sft_items(train, cfg)
        out["sft_test"] = build_sft_items(test, cfg, seed=1)
    if cfg.grpo:
        out["grpo_train"] = build_grpo_items(train, cfg)
        out["grpo_test"] = build_grpo_items(test, cfg, seed=1)
    if cfg.dpo:
        out["dpo_train"] = build_dpo_items(train, cfg)
        out["dpo_test"] = build_dpo_items(test, cfg, seed=1)
    if cfg.multiturn:
        out["multiturn_train"] = build_multiturn_items(train, cfg)
        out["multiturn_test"] = build_multiturn_items(test, cfg, seed=1)
    return out


def _load_dotenv(path=".env"):
    """零依赖地读取工作目录下 `path`（如 .env）填充 os.environ，不覆盖已存在的值。"""
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip().lstrip("\ufeff")  # 容忍文件开头可能的 UTF-8 BOM
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


def main(argv=None):
    _load_dotenv()
    p = argparse.ArgumentParser(description="symbolic regression post-training dataset builder")
    p.add_argument("--sft", action="store_true", dest="sft_set")
    p.add_argument("--grpo", action="store_true", dest="grpo_set")
    p.add_argument("--dpo", action="store_true", dest="dpo_set")
    p.add_argument("--multiturn", action="store_true", dest="multiturn_set")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--dom", type=float, default=10.0)
    p.add_argument("--dist", choices=["uniform", "gaussian"], default="uniform")
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--n-per-group", type=int, default=32)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data")
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--min-depth", type=int, default=4)
    p.add_argument("--max-depth-limit", type=int, default=12)
    # LLM 多轮对话（需配合 --multiturn）；base_url/model/key 优先取 CLI，其次 .env/环境变量
    p.add_argument("--llm", action="store_true", dest="use_llm",
                   help="(多轮) 用 LLM 真对话生成多轮链，需配合 --multiturn")
    p.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", ""))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", ""))
    p.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--llm-max-rounds", type=int,
                   default=int(os.environ.get("LLM_MAX_ROUNDS", 5)))
    p.add_argument("--llm-max-tokens", type=int,
                   default=int(os.environ.get("LLM_MAX_TOKENS", 1024)),
                   help="单次回答 token 上限；推理模型（如 glm-flash）思考链很长时建议提到 4096")
    p.add_argument("--llm-extra-body", default=os.environ.get("LLM_EXTRA_BODY", ""),
                   help="追加到请求体的 JSON（如 GLM 思考强度参数），例如 "
                        '{"thinking":{"type":"enabled","budget_tokens":256}}')
    p.add_argument("--llm-no-fallback", action="store_true",
                   help="LLM 链失败时不回退模板（默认失败自动回退）")
    p.add_argument("--llm-relaxed", action="store_true",
                   help="放宽收敛：模型无法精确符号回归时，以'最佳改进拟合'作为末轮答案，而"
                        "非强制等于真值；需配合 --multiturn --llm 使用")
    p.add_argument("--llm-relaxed-floor", type=float, default=0.4,
                   help="放宽模式下末轮最佳拟合须达到的数值质量下限（默认 0.4）")
    p.add_argument("--llm-relaxed-gain", type=float, default=0.05,
                   help="放宽模式下末轮须比首轮假设提升的数值增量（默认 0.05）")
    p.add_argument("--llm-max-workers", type=int,
                   default=int(os.environ.get("LLM_MAX_WORKERS", 1)),
                   help="LLM 多轮生成并发目标数（>1 用线程池，提吞吐，需 API 支持并发；默认 1）")
    args = p.parse_args(argv)

    cfg = GenerationConfig(
        sft=args.sft_set or not (args.grpo_set or args.dpo_set or args.multiturn_set),
        grpo=args.grpo_set,
        dpo=args.dpo_set,
        multiturn=args.multiturn_set,
        n=args.n, dim=args.dim, max_depth=args.max_depth,
        dom=args.dom, dist=args.dist, group=args.group,
        n_per_group=args.n_per_group, noise=args.noise,
        min_depth=args.min_depth, max_depth_limit=args.max_depth_limit,
        test_ratio=args.test_ratio, seed=args.seed, out_dir=args.out,
        use_llm=args.use_llm,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_max_rounds=args.llm_max_rounds,
        llm_max_tokens=args.llm_max_tokens,
        llm_extra_body=_parse_extra_body(args.llm_extra_body),
        llm_fallback=(not args.llm_no_fallback
                      and os.environ.get("LLM_FALLBACK", "1") != "0"),
        llm_relaxed=args.llm_relaxed,
        llm_relaxed_floor=args.llm_relaxed_floor,
        llm_relaxed_gain=args.llm_relaxed_gain,
        llm_max_workers=args.llm_max_workers,
    )

    rng = random.Random(cfg.seed)
    global seen
    seen = set()
    expr_all = produce_expressions(cfg, rng)
    train, test = split_sets(expr_all, cfg, rng)
    print(f"生成 {len(expr_all)} 个表达式并校验通过；train={len(train)} test={len(test)}")

    datasets = build_all(train, test, cfg)
    os.makedirs(args.out, exist_ok=True)
    for name, items in datasets.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        save_jsonl(items, path)
        print(f"  written {len(items)} 条 -> {path}")

    # 直方：难度分布
    from collections import Counter
    diff = Counter(quality.difficulty(e) for e in expr_all)
    print("难度分布:", dict(diff))


if __name__ == "__main__":
    main()