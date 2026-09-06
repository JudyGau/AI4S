"""symreg_dataset：符号回归 LLM 后训练数据集构建工具。

覆盖 SFT / GRPO / DPO 三种训练范式，方法对齐 SymbArena（arXiv:2508.09897）。
"""
from .config import GenerationConfig
from . import expressions, instructions, normalize, quality, reward
from . import sampling, schemas, symbols

__all__ = [
    "GenerationConfig",
    "expressions",
    "instructions",
    "normalize",
    "quality",
    "reward",
    "sampling",
    "schemas",
    "symbols",
]

__version__ = "0.1.0"