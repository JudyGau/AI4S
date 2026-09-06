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

# 多轮改走 LLM 真对话（OpenAI 兼容 API），API 不可用时自动回退模板
python -m scripts.generate --multiturn --llm --llm-base-url http://127.0.0.1:8000/v1 --llm-model Qwen-7B --llm-api-key sk-xxx --n 300
```

> **便捷脚本**：`python -m scripts.generate_llm --n 100` 已预置调试好的 LLM 多轮链
> 参数（`--multiturn --llm --llm-relaxed` ＋ 深度 4–6 ＋ 至多 8 轮）。它复用同一套表达式
> 管线，仅暴露常用开关（`--n --out --seed --dim --min-depth --max-depth-limit
> --llm-max-rounds --llm-relaxed-floor --llm-relaxed-gain --no-fallback`），
> llm base_url/model/key 优先取命令行、其次 `.env`/环境变量。

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
| `--llm` | flag | — | （需配合 `--multiturn`）多轮改走 LLM 真对话链 |
| `--llm-base-url` | str | "" | OpenAI 兼容 `/chat/completions` 的 base_url，如 `https://api.deepseek.com/v1` |
| `--llm-model` | str | "" | 模型名 |
| `--llm-api-key` | str | `OPENAI_API_KEY` 环境变量 | API 密钥；留空则读环境变量 |
| `--llm-max-rounds` | int | 5 | 收敛循环最大轮数 |
| `--llm-no-fallback` | flag | 关闭 | 勾选后：LLM 链失败时**不**回退模板（默认失败自动回退） |
| `--llm-relaxed` | flag | 关闭 | 放宽收敛：模型无法精确符号回归时，以循环内"最佳改进拟合"作为末轮答案，而非强制等于真值 |
| `--llm-relaxed-floor` | float | 0.4 | 放宽模式下末轮最佳拟合须达到的数值质量下限 |
| `--llm-relaxed-gain` | float | 0.05 | 放宽模式下末轮须比首轮假设提升的数值增量 |
| `--llm-extra-body` | str | `LLM_EXTRA_BODY` | 追加到每次请求体的厂商专属字段（JSON），如 GLM 思考参数；非法 JSON 忽略 |
| `--llm-max-workers` | int | 1 | LLM 多轮生成的**并发目标数**（`>1` 用线程池提吞吐，需 API 支持并发；实测 6 并发约 6× 提速） |

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

## LLM 多轮对话（`--llm`）

如果不满足于模板式的"改系数近邻"，可让**更强的 LLM 真对话**产出分析、假设与依反馈修正的完整轨迹，
再蒸馏进目标模型。这是当前 SR 后训练数据的常见做法。

### 原理：奖励门控收敛

对每个目标表达式，循环以下流程，直到模型输出收敛到 `reference.expr`：

1. **问题轮**：把数据点发成符号回归任务。system 提示「回答最后一行写成 `f = 最简表达式`」。
2. **LLM 假设轮**：模型给出"分析 + 初步假设 `f`"。
3. **奖励门控**：用 `reward.compute_reward` 在**真实数据**上给该候选打数值分 `num`，据此生成**数值反馈**——
   只回 `num`（R²/容差），**绝不透露目标表达式**，避免修正退化成"照着念"。
4. **反馈→再修正**：把反馈作为下一条 human 消息发给模型，循环本轮。
5. **收敛判定**：某轮解析出的表达式与 `reference.expr` 一致即视为收敛，整条链入库。

约束与兜底：

- **首轮即正确 → 丢弃**：没有"不完美中间假设"，不构成多轮修正链。
- **未收敛 / 输出不可解析 / API 异常 → 自动回退模板路径**（除非 `--llm-no-fallback`）。
- 收敛链中的**中间轮**即为真实的不完美假设（由模型产生，语言自然）。

> **放宽收敛（`--llm-relaxed`）**：很多便宜但非专精的模型（如 `glm-4-flash`）只能给出**近似拟合**、
> 无法精确恢复目标闭合式，因而到不了严格的数值收敛门槛。此时可开启 `--llm-relaxed`：
> 循环耗尽后，取所有可解析轮次里**数值质量最高**的那轮作为末轮答案，只要它比首轮假设明显改进
> （≥ `--llm-relaxed-gain`）且达到质量下限（≥ `--llm-relaxed-floor`）即保留为一条**真实的多轮改进链**。
> 代价是末轮答案不再等于 `reference.expr`（`reference` 仍存正确真值，便于评测差距）。若模型连
> 质量下限都达不到（对难度过高的目标很常见），仍会回退模板。

### LLM 路径的数据结构

与模板多轮**完全相同的 schema**（复用 `make_multiturn_item`），只把 `conversations` 换成 LLM 生成的多段：

```json
{
  "conversations": [
    { "from": "human", "value": "进行符号回归：x = [...], y = [...]。……" },
    { "from": "gpt",   "value": "数据呈周期性，初步判断为三角组合。首轮假设 f = 2.0*sin(0.8*x)……" },
    { "from": "human", "value": "候选数值质量≈0.51，明显低于目标。请修正系数与算子，最后一行写 f = 表达式……" },
    { "from": "gpt",   "value": "修正后收敛。f = cos(2.7*exp(3*x))" }
  ],
  "reference": { "expr": "cos(2.7*exp(3*x))", "var_symbols": ["x"],
                 "skeleton": "cos(c0*exp(c1*x))", "difficulty": "medium",
                 "ops": 4, "depth": 5, "vars": 1 }
}
```

要点与代码位置：

- `conversations` 首段必须为 human 问题，末段必须收敛到正确表达式；中间是 human/gpt 交替的反馈-修正轮。
- 角色 `human`/`gpt` 训练时映射为 `user`/`assistant`（与其它范式一致）。
- 实现见 `symreg_dataset/llm.py`（`generate_chain` 奖励门控循环、`_extract_expr` 解析、`make_feedback` 数值反馈），
  提供商抽象在 `symreg_dataset/providers.py`（`OpenAiCompatible`，仅标准库 urllib，零第三方依赖）。
- 中途失败由 `symreg_dataset/builders/multiturn.py::build_multiturn_items` 自动回退模板。

### 配置方式：`.env` 一键接入

`scripts/generate.py` 启动时会自动读取项目根目录的 `.env`（若存在），把其中条目注入环境变量，
且 **不覆盖** `--llm-*` 命令行参数已显式给定的值。优先级：CLI > 环境变量 > `.env`。

参考模板见 `.env.example`，常用项：

```bash
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4    # OpenAI 兼容端点
OPENAI_API_KEY=sk-xxx                               # 密钥
LLM_MODEL=glm-4.7-FlashX                            # 模型名
LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}     # 厂商专属字段（可选）
LLM_MAX_ROUNDS=6                                    # 可选，默认 5
```

> 实现上 `LLM_EXTRA_BODY` 会原样合并进每次请求体。很多"强制思考"模型平时不直接吐正文，
> 实测在智谱 `glm-4.7(-FlashX)` 上设 `{"thinking":{"type":"disabled"}}` 可关闭思考、让 `content`
> 正文直出且响应秒回——这是它能被本流水线所用的关键开关（不同厂商字段名不同，按模型文档填）。
> 注意 `.env` 含密钥、已被 gitignore，**不要提交**。

### 大规模并发

LLM 多轮是顺序 API × 多轮，单目标开销明显。要批量生成（几百~上千条），用并发摊薄：
`--llm-max-workers N`（配合线程池，每个线程一个目标）。实测 100 目标、`--llm-max-workers 6`
约 10 分钟内完成（串行 60 目标曾 ~35 分钟），**吞吐约 6×**，且真实 LLM 链产出率 ~69% 稳定不变。
批量建议：

```bash
python -m scripts.generate --multiturn --llm --llm-relaxed \
  --llm-max-rounds 6 --llm-max-workers 6 --n 500 --out data
```

产出的真实 LLM 链占比约 7 成，余下自动回退模板兜底；多数真实链末轮收敛到 `reference.expr`。

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
- **多轮（multiturn）**：模板路径下 `conversations` 共 4 段（分析+假设/反馈/收敛）；`--llm` 时用真对话链（其间可能有额外反馈-修正轮）。中间假设的语义与数据形态见上文两个专属小节。

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
| `use_llm` / `llm_*` | 见下 | LLM 多轮的运行时配置；CLI 未暴露 `llm_timeout`（30s）与 `llm_temperature`（0.7），可在 `config.py` 调整 |

## 与训练的衔接

GRPO 用到的奖励函数示例见 `scripts/integrate_reward.py`：它按 `grpo_*.jsonl` 中
每条的 `reference` + `data`，用 `reward.compute_reward` 对模型输出打分
（数值项 R²/Accτ + 形式一致性项），可直接对应 LLaMA-Factory 的自定义 reward 函数入口。

## 可复现性说明

- 设置相同的 `--seed`，生成结果确定（表达式、采样、划分、指令措辞均确定）。
- 数据点与表达式常数统一四舍五入到 `precision`（默认 2）位小数，确保写出内容可复现、可被模型稳定学习。