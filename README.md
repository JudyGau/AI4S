# 符号回归 LLM 后训练数据集构建（symreg_dataset）

用于为**符号回归（symbolic regression）**基座大模型生成后训练数据集的 Python 工具。
方法对齐 **SymbArena / Symbolic-R1（arXiv:2508.09897）**。

覆盖三种训练范式，统一输出 **JSONL**：
- **SFT** 监督微调：`instruction -> 目标表达式`
- **Form-GRPO** 强化学习：问题式输入 + reference/data（供奖励函数打分）
- **DPO** 偏好对齐：同一指令下 `chosen`（正确）/ `rejected`（扰动负样本）

## 特性
- 树式随机生成一/多元数学表达式（SymPy），支持算子/终结符结构约束。
- 表达式**骨架（系数抽象）**用于结构级唯一性去重、形式一致性奖励与 train/test 防泄露划分。
- 数据矩阵采样：`U(-dom,dom)` / `N(0,dom)`，多组、多变量、可选高斯噪声，自动过滤非有限值。
- 数值奖励（RMSE / R² / Acc_τ）+ 形式一致性奖励（规范化字符串相似度）。
- 复杂度过滤（树深 4~12）、难度分级（easy/medium/hard）、骨架唯一性。
- 内置真实表达式库（Feynman 代表式 + Nguyen 基准），可选做"现实性"筛选。
- 对齐 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的 JSONL 约定。

## 安装
```bash
pip install -r requirements.txt
```

## 快速开始
```bash
# 生成三种范式的数据（默认 100 条，含 train/test 划分）
python -m scripts.generate --n 100 --dim 1

# 仅 SFT + DPO，2 变量，带噪声，输出到 data/
python -m scripts.generate --sft --dpo --n 500 --dim 2 --noise 0.05 --out data
```

输出文件：`data/{sft,grpo,dpo}_{train,test}.jsonl`

> 生成脚本的完整参数说明、输出格式、质量过滤与训练衔接详见 **[docs/generate.md](docs/generate.md)**。

## 单元测试
```bash
python -m pytest tests/
```

## 主要模块
| 模块 | 说明 |
|---|---|
| `expressions.py` | 随机方程生成、骨架抽取 |
| `normalize.py` | 表达式规范化 / 稳定序列化 / 解析 |
| `sampling.py` | 数据矩阵采样 |
| `quality.py` | 复杂度、难度、唯一性、train/test 划分 |
| `reward.py` | 数值 + 形式奖励 |
| `real_lib.py` | 真实表达式库与"现实性"筛选 |
| `schemas.py` | LLaMA-Factory 兼容字段 |
| `builders/` | SFT / GRPO / DPO 构造器 |
| `scripts/integrate_reward.py` | GRPO 自定义奖励接入示例 |

## 数据流
```
随机方程(树式) -> 质量过滤(复杂度/唯一性/数值稳定/现实性)
   -> 按骨架划分 train/test
   -> 采样数据矩阵 -> 指令模板封装
   -> SFT / GRPO / DPO JSONL
```

## 文献参考
Hua et al., *Finetuning Large Language Model as an Effective Symbolic Regressor*（SymbArena / Symbolic-R1）, arXiv:2508.09897.