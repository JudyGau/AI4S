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

    # LLM 多轮对话（OpenAI 兼容 API）
    use_llm: bool = False          # 多轮是否走 LLM 真对话（需配置下面连接信息）
    llm_base_url: str = ""         # 例如 https://api.deepseek.com/v1 或 https://api.openai.com/v1
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout: float = 30.0      # 单次请求超时（秒）
    llm_max_rounds: int = 5        # 收敛循环最多轮次
    llm_temperature: float = 0.7   # 采样温度
    llm_max_tokens: int = 1024     # 单次回答 token 上限（推理模型思考链会占用预算，需给足）
    llm_extra_body: dict | None = None  # 追加到请求体的厂商专属字段（如 GLM 思考强度），JSON 解析
    llm_fallback: bool = True      # 链失败（API 错误/未收敛/首轮即对）时回退模板路径
    llm_relaxed: bool = False      # 放宽"末轮=精确解"：模型无法精确符号回归时，接受"最佳拟合"作末轮答案
    llm_relaxed_floor: float = 0.4  # 放宽模式下末轮最佳拟合须 ≥ 该数值质量下限
    llm_relaxed_gain: float = 0.05  # 放宽模式下末轮须比首轮假设提升 ≥ 该数值增量
    llm_max_workers: int = 1     # LLM 多轮生成并发数（=并发目标数，提吞吐；>1 需 API 支持并发）