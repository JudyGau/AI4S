# 生成脚本说明

`scripts/generate.py` 是数据集生成的唯一入口。它完成：随机生成方程 → 质量过滤 →
按骨架划分 train/test → 采样数据矩阵 → 指令封装 → 写出 SFT / GRPO / DPO 三种 JSONL。

## 运行方式

```bash
# 工作目录为项目根（含 symreg_dataset 包），直接以模块方式运行
python -m scripts.generate [选项]
```

用 `--help` 查看全部选项：

```bash
python -m scripts.generate --help
```

## 常用示例

```bash
# 默认：仅 SFT，100 条，单变量，输出到 data/
python -m scripts.generate

# 三种范式全开，双变量，加 5% 高斯噪声
python -m scripts.generate --sft --grpo --dpo --n 500 --dim 2 --noise 0.05

# 仅生成 DPO，300 条，多点（每组 64 点）
python -m scripts.generate --dpo --n 300 --n-per-group 64

# 高斯分布采样自变量，控制随机种子，输出到自定义目录
python -m scripts.generate --sft --grpo --dist gaussian --seed 42 --out mydata
```

## 参数说明

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--sft` / `--grpo` / `--dpo` | flag | 都不传时默认只开 SFT | 要生成的范式。显式传了其中任意一个，则未指定的范式不会生成 |
| `--n` | int | 100 | 每个范式按**通过质量过滤的表达式数**计的目标样本量 |
| `--dim` | int | 1 | 自变量维度（多变量回归） |
| `--max-depth` | int | 4 | 随机表达式树的**最大**递归深度（树过大，见 `--max-depth-limit` 限制） |
| `--dom` | float | 10.0 | 自变量采样域半径，变量取值约在 ±dom |
| `--dist` | uniform/gaussian | uniform | 自变量分布：均匀 U(-dom, dom) 或高斯 N(0, dom) |
| `--group` | int | 1 | 每条样本含的数据组数（多组数据更接近多任务） |
| `--n-per-group` | int | 32 | 每组数据点数 |
| `--noise` | float | 0.0 | 加到因变量上的高斯噪声标准差 |
| `--seed` | int | 0 | 随机种子（保证可复现） |
| `--out` | str | data | 输出目录 |
| `--test-ratio` | float | 0.1 | test 占全部**骨架**的比例（按骨架划分，防形式泄露） |
| `--min-depth` | int | 4 | 复杂度下限：树深低于此值的表达式被过滤 |
| `--max-depth-limit` | int | 12 | 复杂度上限：树深高于此值的表达式被过滤 |

### 关于三种范式的开关语义

- 三个 flag 都不传 → 只生成 **SFT**。
- 传了任意一个或多个 → 只生成被选中的范式。
例如 `--grpo` 单独执行只生成 GRPO 数据。

## 输出文件

在 `--out` 目录下，每个启用的范式输出 train 与 test 两个 JSONL：

```
data/
├── sft_train.jsonl   sft_test.jsonl    # SFT：单轮 human -> gpt
├── grpo_train.jsonl  grpo_test.jsonl   # GRPO：问题式 human + reference + data
└── dpo_train.jsonl   dpo_test.jsonl    # DPO：instruction + chosen / rejected
```

train/test 按**骨架**（系数抽象后的规范形式）划分，保证同一形式（不同系数）不会横跨两侧。

### 各范式字段速览（对齐 LLaMA-Factory）

- **SFT**：`system` + `conversations: [human, gpt]`，其中 gpt 为目标表达式，另附 `reference` 元数据（expr / skeleton / difficulty / 复杂度）。
- **GRPO**：`conversations: [human]`（只有问题，无标准答案），附 `reference`（目标表达式字符串）与 `data`（分组数据矩阵、dim、分布），供自定义 reward 环境打分。
- **DPO**：`instruction` + `chosen`（正确表达式）+ `rejected`（扰动/错误表达式），构造时已用 `reward.compute_reward` 校验 chosen 严格优于 rejected。

## 质量过滤流程

生成的每个表达式需全部通过以下过滤才会进入样本：

1. 不含符号无穷/NaN 等病态节点（`sp.zoo/nan/oo`）。
2. 树深在 `[min-depth, max-depth-limit]` 区间（默认 4~12）。
3. 骨架唯一（结构级去重，避免近似重复样本）。
4. 采样数值稳定：所有点为有限值，且因变量量级不超过 `config.max_abs`（默认 1e4，过滤 `exp(exp(·))` 这类天文量级）。
5. 现实性软过滤（默认开启但阈值为 0，仅做基本结构校验；见 config）。

若在 `n` 次目标内无法产生足够通过过滤的表达式，会抛错并提示放宽参数
（例如提高 `--max-depth`、调大 `--dom`，或减小 `--min-depth`/`--max-depth-limit` 区间）。

## 未暴露到 CLI 的配置项

CLI 覆盖了大多数常用项；以下几个仅能通过修改 `symreg_dataset/config.py` 的 `GenerationConfig` 调整：

| 字段 | 默认 | 说明 |
|---|---|---|
| `max_abs` | 1e4 | 因变量量级上限（质量过滤第 4 步） |
| `use_real_lib` | True | 是否启用真实表达式库（Feynman/Nguyen）自相似度软过滤 |
| `real_lib_threshold` | 0.0 | 现实性阈值，>0 时会据此过滤与已知科学方程差异过大的畸形表达式 |
| `precision` | 2 | 数据点与表达式常数的保留小数位 |
| `system` | 科学助手提示词 | 写入样本的 system 字段 |

## 与训练的衔接

GRPO 用到的奖励函数示例见 `scripts/integrate_reward.py`：它按 `grpo_*.jsonl` 中
每条的 `reference` + `data`，用 `reward.compute_reward` 对模型输出打分
（数值项 R²/Accτ + 形式一致性项），可直接对应 LLaMA-Factory 的自定义 reward 函数入口。

## 可复现性说明

- 设置相同的 `--seed`，生成结果确定（表达式、采样、划分、指令措辞均确定）。
- 数据点与表达式常数统一四舍五入到 `precision`（默认 2）位小数，确保写出内容可复现、可被模型稳定学习。