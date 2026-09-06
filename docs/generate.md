# 生成脚本说明

`scripts/generate.py` 是数据集生成的唯一入口。它完成：随机生成方程 → 质量过滤 →
按骨架划分 train/test → 采样数据矩阵 → 指令封装 → 写出 SFT / GRPO / DPO / 多轮四种 JSONL。

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

# 综合多轮链（problem -> 分析+假设 -> 反馈 -> 最终表达式），2 ~ 3 轮问答
python -m scripts.generate --multiturn --n 300
```

## 参数说明

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--sft` / `--grpo` / `--dpo` / `--multiturn` | flag | 都不传时默认只开 SFT | 要生成的范式。显式传了其中任意一个，则未指定的范式不会生成 |
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

### 关于各范式的开关语义

- 四个 flag 都不传 → 只生成 **SFT**。
- 传了任意一个或多个 → 只生成被选中的范式。
例如 `--grpo` 单独执行只生成 GRPO 数据；`--multiturn` 单独执行只生成多轮数据。

## 多轮对话：生成逻辑与数据结构

`--multiturn` 生成的是**综合分析链**——让模型先提出一个中间假设、再依据反馈修正、最终收敛到正确答案，
用于训练它掌握「先分析假设、依反馈自我修正」的多步推理能力，这与 Symbolic-R1 的
Hypothesis-Experiment-Revision（HER）思路一致。

### 生成逻辑（每条样本的构造步骤）

对每个通过质量过滤的目标表达式 `f*`，按以下流程构造一条多轮样本：

1. **采样数据点**：对 `f*` 按 `--dist` / `--group` / `--n-per-group` / `--noise` 采样数据矩阵。
2. **问题轮（human）**：用 SFT 指令模板把数据点格式化成符号回归任务（与单轮 SFT 相同表达）。
3. **分析 + 假设轮（gpt）**：
   - 从 `REASON_PHRASES` 随机选一条定性分析（如「周期/振荡」「幂/多项式趋势」等）；
   - 用 `_perturb(keep_close=True)` 生成**结构近邻**的候选（改单系数 / sin↔cos 换算子，
     必要时整体小幅缩放平移，**排除 -expr 大幅翻转**）；
   - 在采样数据上用 `reward` 的**数值项 num** 打分，把候选约束在质量区间内：
     - 上限 `target_num - ε`：排除数值上≈目标的候选（修正无意义）；
     - 下限 `0.25 × target_num`：排除完全乱猜的候选（如换算子直接毁掉函数）；
     - 取区间内 **num 最高**的候选作为 `f1`（最贴近目标的可信"首猜"）；
   - 保证 `f1` 的规范字符串与最终式不同（最多重试 100 次并做字符串去重）；
   - 若连一个合理假设都取不到（如 `log(Abs(x))` 这类无参数、学家系数后必毁函数的病态式），
     跳过该样本，不强行生成"乱猜"假设。
   - 拼成「初步分析 + f1 = <候选>」作为第一段助手回复。
4. **反馈轮（human）**：随机选一条反馈模板，指出 `f1` 拟合不佳并要求修正到最终表达式。
5. **收敛轮（gpt）**：输出**正确的**最终表达式 `f*`（与 `reference.expr` 完全一致）。

> 质量约束的作用：频域/系数改动是"大杠杆"，小幅扰动就会显著降低数值拟合，因此下限设为
> `0.25 × target_num` 而非更严阈值，以保证总能选出"虽不完美但可信"的首猜；`_perturb` 通过
> 可选的 `keep_close` 参数与 DPO 的硬负样本（含 `-expr` 翻转）解耦，两种负样本各取所需。

### 数据结构

单条样本（JSON 对象）：

```json
{
  "system": "你是擅长符号回归的科学助手。",
  "conversations": [
    { "from": "human", "value": "进行符号回归：x1 = [...], y = [...]。请给出函数关系……" },
    { "from": "gpt",   "value": "初步观察，数据大致呈指数增长/衰减特征。我先抛出一个中间候选 f1 = sin(2.7*exp(3*x))……" },
    { "from": "human", "value": "初步假设还不够贴合数据……请给出最终表达式 f = ..." },
    { "from": "gpt",   "value": "cos(2.7*exp(3*x))" }
  ],
  "reference": { "expr": "cos(2.7*exp(3*x))", "var_symbols": ["x"],
                 "skeleton": "cos(c0*exp(c1*x))", "difficulty": "medium",
                 "ops": 4, "depth": 5, "vars": 1 }
}
```

要点：

- `conversations` 固定为 **4 段**、交替出现：`human → gpt → human → gpt`。
- 角色沿用 `human`/`gpt`（LLaMA-Factory 的 sharegpt 约定），训练时映射为 `user`/`assistant`。
- 第 2 段（分析+假设）包含候选 `f1`，是「非最终」的中间态；**第 4 段才是正确答案** `reference.expr`。
- `reference` 元数据与 SFT 完全一致（表达式/骨架/难度/复杂度），便于校验多轮收敛、评估或进阶构造 GRPO 奖励。
- 中间假设 `f1` 与最终式不同，保证每一轮对话都有真实的"修正"语义，而非冗余重复。

## 输出文件

在 `--out` 目录下，每个启用的范式输出 train 与 test 两个 JSONL：

```
data/
├── sft_train.jsonl   sft_test.jsonl       # SFT：单轮 human -> gpt
├── grpo_train.jsonl  grpo_test.jsonl      # GRPO：问题式 human + reference + data
├── dpo_train.jsonl   dpo_test.jsonl       # DPO：instruction + chosen / rejected
└── multiturn_train.jsonl  multiturn_test.jsonl  # 多轮：分析+假设 -> 反馈 -> 最终表达式
```

train/test 按**骨架**（系数抽象后的规范形式）划分，保证同一形式（不同系数）不会横跨两侧。

### 各范式字段速览（对齐 LLaMA-Factory）

- **SFT**：`system` + `conversations: [human, gpt]`，其中 gpt 为目标表达式，另附 `reference` 元数据（expr / skeleton / difficulty / 复杂度）。
- **GRPO**：`conversations: [human]`（只有问题，无标准答案），附 `reference`（目标表达式字符串）与 `data`（分组数据矩阵、dim、分布），供自定义 reward 环境打分。
- **DPO**：`instruction` + `chosen`（正确表达式）+ `rejected`（扰动/错误表达式），构造时已用 `reward.compute_reward` 校验 chosen 严格优于 rejected。
- **多轮（multiturn）**：`conversations` 共 4 段——`human`(数据点问题) → `gpt`(分析 + 中间假设 f1) → `human`(反馈要求修正) → `gpt`(收敛后的最终正确表达式)。中间假设由 `_perturb(keep_close=True)` 生成**结构近邻**候选，并在数值质量区间 `[0.25×target_num, target_num-ε)` 内取 `num` 最高的作为 `f1`，用于训练模型「先分析假设、依反馈自我修正」。

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