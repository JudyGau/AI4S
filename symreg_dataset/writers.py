"""JSONL 写出 + 去重。"""
from __future__ import annotations

import gzip
import json
import os
from typing import Iterable


def save_jsonl(items: Iterable[dict], path: str, append: bool = False) -> str:
    """把 dict 列表写入 JSONL；若 path 以 .gz 结尾则压缩写入。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "at" if append else "wt"
    if path.endswith(".gz"):
        opener = gzip.open(path, mode, encoding="utf-8")
    else:
        opener = open(path, mode, encoding="utf-8")
    with opener as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path


def dedup_by_key(items: list[dict], key_fn) -> list[dict]:
    seen = set()
    out = []
    for it in items:
        k = key_fn(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out