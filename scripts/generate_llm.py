"""便捷脚本：以调试好在真实模型上能收敛的参数组合，生成 LLM 多轮链数据集。

等价于：
    python -m scripts.generate --multiturn --llm --llm-relaxed \
        --min-depth 4 --max-depth-limit 6 --llm-max-rounds 8 --n 100
仅暴露少量常用开关；llm base_url/model/key 优先取命令行，其次 .env/环境变量。

用法：
    python -m scripts.generate_llm --n 100
    python -m scripts.generate_llm --n 50 --seed 42 --out data/llm_medium
"""
from __future__ import annotations

import argparse
import os
import random

from scripts.generate import (_load_dotenv, _parse_extra_body,
                              produce_expressions, split_sets, build_all, seen)
from symreg_dataset import quality
from symreg_dataset.config import GenerationConfig


def main(argv=None):
    _load_dotenv()
    p = argparse.ArgumentParser(
        description="生成 LLM 多轮链数据集（默认 multiturn+llm+relaxed，中等难度）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="目标表达式数量")
    p.add_argument("--out", default="data", help="输出目录")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--min-depth", type=int, default=4, help="树深下限")
    p.add_argument("--max-depth-limit", type=int, default=6, help="树深上限（中等难度推荐 4–6）")
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--n-per-group", type=int, default=32)
    p.add_argument("--dom", type=float, default=10.0)
    p.add_argument("--dist", choices=["uniform", "gaussian"], default="uniform")
    p.add_argument("--noise", type=float, default=0.0)
    # LLM 相关
    p.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", ""))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", ""))
    p.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--llm-max-rounds", type=int, default=8, help="单条链最多轮数")
    p.add_argument("--llm-max-tokens", type=int,
                   default=int(os.environ.get("LLM_MAX_TOKENS", 1024)))
    p.add_argument("--llm-extra-body", default=os.environ.get("LLM_EXTRA_BODY", ""),
                   help='追加到请求体的 JSON，如 {"thinking":{"type":"disabled"}}')
    p.add_argument("--llm-relaxed-floor", type=float, default=0.4)
    p.add_argument("--llm-relaxed-gain", type=float, default=0.05)
    p.add_argument("--no-fallback", action="store_true",
                   help="LLM 链失败时不回退模板（默认自动回退）")
    args = p.parse_args(argv)

    cfg = GenerationConfig(
        sft=False, grpo=False, dpo=False, multiturn=True,
        n=args.n, dim=args.dim, min_depth=args.min_depth,
        max_depth_limit=args.max_depth_limit, test_ratio=args.test_ratio,
        n_per_group=args.n_per_group, dom=args.dom, dist=args.dist,
        noise=args.noise, seed=args.seed, out_dir=args.out,
        use_llm=True, llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key, llm_model=args.llm_model,
        llm_max_rounds=args.llm_max_rounds, llm_max_tokens=args.llm_max_tokens,
        llm_extra_body=_parse_extra_body(args.llm_extra_body),
        llm_fallback=not args.no_fallback,
        llm_relaxed=True,
        llm_relaxed_floor=args.llm_relaxed_floor,
        llm_relaxed_gain=args.llm_relaxed_gain,
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
        from symreg_dataset.writers import save_jsonl
        save_jsonl(items, path)
        print(f"  written {len(items)} 条 -> {path}")

    from collections import Counter
    diff = Counter(quality.difficulty(e) for e in expr_all)
    print("难度分布:", dict(diff))


if __name__ == "__main__":
    main()