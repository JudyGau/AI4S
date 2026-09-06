"""生成配置。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    # 范式开关
    sft: bool = True
    grpo: bool = True
    dpo: bool = True
    multiturn: bool = False    # 综合多轮链（problem->分析->反馈->最终表达式）

    # 规模
    n: int = 100                 # 每个范式生成的目标样本数（按表达式计数）
    dim: int = 1                 # 自变量维度
    max_depth: int = 4           # 表达式树最大深度

    # 数据矩阵
    dom: float = 10.0
    dist: str = "uniform"        # uniform | gaussian
    group: int = 1               # 每组方程的数据组数
    n_per_group: int = 32        # 每组数据点数
    noise: float = 0.0           # 高斯噪声标准差

    # 质量控制
    min_depth: int = 4
    max_depth_limit: int = 12
    max_abs: float = 1e4           # 因变量量级上限：超出的表达式视为数值病态，过滤
    use_real_lib: bool = True    # 生成时允许以真实库自相似度做软过滤
    real_lib_threshold: float = 0.0  # 0 表示仅做基本结构过滤

    # train/test
    test_ratio: float = 0.1

    # 其他
    precision: int = 2
    seed: int = 0
    out_dir: str = "data"
    system: str = "你是擅长符号回归的科学助手。"