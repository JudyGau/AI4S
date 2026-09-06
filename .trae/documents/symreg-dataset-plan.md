# 符号回归 LLM 后训练数据集构建项目 — 实施方案

## Context（背景）

从零构建一个**符号回归（symbolic regression, SR）**的 LLM 后训练数据集项目。核心思想：
把每个样本表示为「数据矩阵 d_i + 目标方程 f_i」，即随机生成方程（树式生成），采样多组数据点，
再按指令模板封装，产出可直接用于 **SFT / GRPO / DPO** 三种训练范式的 JSONL。

方法对齐参考文献 **SymbArena / Symbolic-R1（arXiv:2508.09897）**：
- 树式方程生成 + 算子/终结符符号库 + 结构约束。
- 骨架（coefficient-abstracted）唯一性校验与复杂度过滤（树深 4~12）。
- data matrix：均匀 U(−dom,dom) 或高斯 N(0,dom) 采样，dom=10，多变量、多组。
- 按骨架划分 train/test 防形式泄露。
- 评估：数值（R² / Acc_τ）+ 形式一致性（canonical 形式的子串/序列相似度）。
- SFT → Form-GRPO（结构感知的形式奖励 + 数值奖励）→ DPO。

环境：`d:\trae_projects\AI4S` 为空目录（全新项目）。技术栈 **Python + SymPy**，产物 **JSONL**。

## 项目结构

```
d:\trae_projects\AI4S\
├── symreg_dataset/                 # 核心包
│   ├── __init__.py
│   ├── symbols.py                  # 算子/终结符符号库 + 结构约束
│   ├── expressions.py              # 多变量随机方程（树式生成）+ 参数/骨架抽取
│   ├── normalize.py                # 表达式规范化：canonical化、系数抽象、字符串序列化
│   ├── sampling.py                 # 数据矩阵采样：多变量、多组、噪声（uniform/gaussian）
│   ├── instructions.py             # 多套指令模板（多变量场景）
│   ├── quality.py                  # 复杂度校验、骨架唯一性、按骨架划分 train/test
│   ├── reward.py                   # 数值奖励(R²/RMSE/Accτ) + 形式一致性奖励
│   ├── real_lib.py                 # 真实表达式库（Feynman 等）导入 + 现实性筛选
│   ├── schemas.py                  # 对齐 LLaMA-Factory 的 JSONL schema
│   ├── builders/
│   │   ├── __init__.py
│   │   ├── sft.py                  # SFT
│   │   ├── grpo.py                 # Form-GRPO
│   │   └── dpo.py                  # DPO（chosen/rejected）
│   ├── writers.py                  # JSONL 写出 + 去重
│   ├── utils.py
│   └── config.py                   # GenerationConfig
├── scripts/
│   ├── generate.py                 # CLI 总入口
│   └── integrate_reward.py         # GRPO reward 接入示例（LLaMA-Factory/自定义）
├── data/                           # 运行时生成输出（gitignore）
├── tests/
│   ├── test_expressions.py
│   ├── test_normalize.py
│   ├── test_sampling.py
│   ├── test_quality.py
│   ├── test_reward.py
│   └── test_builders.py
├── requirements.txt
└── README.md
```

## 关键设计

### 1. symbols.py — 算子/终结符符号库
按文献定义符号空间并施加结构约束：
- 一元算子：`sin, cos, tan, exp, log, abs, sqrt, ...`；二元算子：`+,-,*,/,**`；常数/参数符号。
- 记录每类算子的合法子节点数，从源头避免非法表达式（如非对称括号、除零、负数开根号等）。

### 2. expressions.py — 多变量方程生成 + 骨架抽取
- `random_equation(dim, max_depth, ops, rng, ...) -> sympy.Expr`：递归增量构造树，
  叶子取自变量占位符 `x_1..x_{dim}` 或常数；支持**多变量**（dim≥1）。
- `skeletonize(expr) -> (skeleton_symbolic, param_map)`：把数值常数替换为自由参数得到
  「表达式骨架」（如 `c0*sin(c1*x1) + c2*x2`），即用户要的"表达式骨架"，也是形式唯一性/形式奖励的依据。

### 3. normalize.py — 表达式规范化
- `canonicalize(expr) -> str`：SymPy simplify + 统一括号/算子顺序 + 排序项 → 标准不依赖系数字面量的规范形式（系数抽象成占位符）。
- `serialize(expr) -> str`：稳定、可复现的字符串序列化（固定精度小数，如 2~3 位）。
- 供 DPO 选定/拒绝、GRPO 形式奖励、以及形式一致性度量的基础。

### 4. sampling.py — 数据矩阵采样
- `sample_data(expr, group, n_per_group, dist, dom, noise, seed) -> data_matrix`
- 同文献：随机取 group 组数据，每组含 dim 个自变量 + 1 个因变量；分布可选 `uniform U(-dom,dom)` 或 `gaussian N(0,dom)`，dom 默认 10；可选噪声。
- 过滤 NaN/Inf 与数值不稳定取值，并校验表达式确实依赖自变量（排除平凡常数）。

### 5. instructions.py — 多套指令模板
多变量回归的中文指令模板（SFT/DPO/GRPO 各自版式），点位按固定精度格式化保证可复现；
提供多套措辞模板以增强数据多样性。

### 6. quality.py — 难度与质量控制（文献 Step D）
- **复杂度过滤**：树深扫描（层数在 4~12 之外过滤）+ 终结符/算子数统计。
- **唯一性**：骨架规范化后做重复检测，保证结构级唯一，避免近似重复样本。
- **难度分级**：按复杂度/结构打 easy/medium/hard 标签（用于进阶采样）。
- **train/test 划分**：按骨架划分，**防止同一形式不同系数的泄露**。

### 7. reward.py — 奖励函数（数值 + 形式）
- 数值：`r2 / rmse / acc_tau(pred, ref, points)`（拟合保真度）。
- 形式：`form_similarity(pred_canonical, ref_canonical)` 基于规范化系数抽象形式的串相似度
  （文献 heuristics 形式一致性度量的轻量实现）。
- Form-GRPO 奖励 = 数值项 + 形式项的组合。DPO 用其校验 chosen 严格优于 rejected。

### 8. real_lib.py — 真实表达式库 + 现实性筛选（文献 reality enhancement）
- 内置/可导入已知科学方程库（如 **Feynman Symbolic Regression Database / AIFeynman** 的 100 个物理方程，
  以及 Nguyen、R-rational 等）作为种子集。
- 提供相似度筛选：以生成的方程与库中方程做 skeleton/Jaccard 相似度，过滤"不符合现实"的畸形方程；
  可选接入 LLM 检索辅助筛选（文献做法，默认本地启发式）。

### 9. builders/ 与 schemas.py — 三范式构造器（对齐 LLaMA-Factory）
- `schemas.py` 定义统一字段（system/instruction/input/meta/reference/data/reward），
  并导出 LLaMA-Factory 兼容格式：
  - **SFT**：`{"system","conversations":[{"from":"human","value":...},{"from":"gpt","value":<f(x)=...>}]}`
  - **DPO**：同上 + `"chosen" / "rejected"`（同 instruction 下正确 vs 扰动/过度拟合表达式）
  - **GRPO**：问题式 `conversations`（仅 human 提示，无标准答案）+ 附带 `reference` 与 `data`
    （供自定义 reward 环境用 reward.py 打分），并随附 `scripts/integrate_reward.py` 接入示例。

### 10. config.py + scripts/generate.py — 配置与 CLI
- `GenerationConfig` dataclass：范式选择、样本量、dim、算子集、dom、组数/每组点数、噪声、
  复杂度区间、难度比例、真实库开关、输出路径、seed。
- CLI：`python -m scripts.generate --sft --grpo --n 100 --dim 2 --out data/`，一步导出多范式 JSONL。

## 验证方式
- `python -m pytest tests/` 全绿。
- 运行 `python -m scripts.generate --n 20 --dim 1`，核对输出 `data/sft/dpo/grpo.jsonl`：
  - 条目数与配置一致；指令含规范化的多组数据点；
  - 每个目标表达式可被 SymPy 解析、无 NaN/Inf，且在数据上 R² 接近 1（保证数据与表达式自洽）；
  - 骨架唯一性校验通过；train/test 无同骨架样本；
  - DPO 所有 chosen 的形式/数值奖励严格优于 rejected。