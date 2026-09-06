"""CLI 总入口：一次生成 SFT / GRPO / DPO 三种范式的 JSONL 数据集。

用法示例：
    python -m scripts.generate --n 20 --dim 1
    python -m scripts.generate --sft --dpo --n 100 --dim 2 --noise 0.05 --out data
"""
from __future__ import annotations

import argparse
import os
import random

import sympy as sp

from symreg_dataset.config import GenerationConfig
from symreg_dataset import expressions, quality, sampling, real_lib
from symreg_dataset.builders import (
    build_sft_items, build_grpo_items, build_dpo_items,
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
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="symbolic regression post-training dataset builder")
    p.add_argument("--sft", action="store_true", dest="sft_set")
    p.add_argument("--grpo", action="store_true", dest="grpo_set")
    p.add_argument("--dpo", action="store_true", dest="dpo_set")
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
    args = p.parse_args(argv)

    cfg = GenerationConfig(
        sft=args.sft_set or not (args.grpo_set or args.dpo_set),
        grpo=args.grpo_set,
        dpo=args.dpo_set,
        n=args.n, dim=args.dim, max_depth=args.max_depth,
        dom=args.dom, dist=args.dist, group=args.group,
        n_per_group=args.n_per_group, noise=args.noise,
        min_depth=args.min_depth, max_depth_limit=args.max_depth_limit,
        test_ratio=args.test_ratio, seed=args.seed, out_dir=args.out,
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