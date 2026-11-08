# Qwen3 招聘领域 SFT / DPO 微调项目

本项目使用 LLaMA-Factory 完成招聘领域模型的 SFT 和 DPO，并覆盖数据检查、
防泄漏切分、LoRA 训练、业务评测、模型合并、API 部署和服务压测。

- 教学演示模型：`../../环境和物料/models/models/Qwen--Qwen3-1.7B/snapshots/master`
- 低成本验证模型：`model_files/Qwen3-4B`
- 正式业务模型：`model_files/Qwen3-8B`
- 原始微调语料：`code/data/gemini_labeled_v2/tasks_fixed`
- 训练框架：LLaMA-Factory 0.9.4
- Python 环境：由 `uv` 管理，所有命令都使用 `uv run`

仓库已有训练记录和当前配置针对 **Qwen3-4B** 和 **Qwen3-8B**。

**Qwen3-1.7B** 用于教学演示（Mac CPU/MPS 可跑通流程），配置位于 `code/configs/qwen3_1.7b_*.yaml`。

**Qwen3-4B** 用来快速验证租赁 GPU 环境、LLaMA-Factory 链路、语料可训练性以及相对基座的指标增益；4B 通过不能替代 8B 的显存和业务验收。

**Qwen3-8B** 是正式业务模型，必须独立完成 SFT、DPO 和冻结集评测。
本项目的结构化抽取/分类语料不包含思维链，8B 的 SFT、DPO、评测和 API 统一关闭思考模式；
详细设置与检查命令见 8B 服务器操作指南。

## 操作文档

| 文档 | 用途 |
| --- | --- |
| [`doc/FINETUNING_RUNBOOK.md`](doc/FINETUNING_RUNBOOK.md) | 总流程、SFT/DPO 关系、4B/8B 放行门禁 |
| [`doc/SFT_FULL_PIPELINE.md`](doc/SFT_FULL_PIPELINE.md) | Qwen3-4B SFT 快速验证流程 |
| [`doc/DPO_FINE_TUNING.md`](doc/DPO_FINE_TUNING.md) | Qwen3-4B DPO 快速验证流程 |
| [`doc/QWEN3_8B_FINE_TUNING.md`](doc/QWEN3_8B_FINE_TUNING.md) | Qwen3-8B 服务器完整操作指南 |
| [`doc/CORPUS_PREPARATION.md`](doc/CORPUS_PREPARATION.md) | SFT/DPO 语料准备原则与发布清单 |

## Codex Direct v2 DPO 微调

当前 DPO 发布版包含 900 条训练数据和 100 条偏好评测数据，已注册到
`code/data/datasets/llamafactory/dataset_info.json`。先将完成的 SFT Adapter 合并成 DPO 起点模型，再启动 DPO：

```bash
uv run llamafactory-cli export \
  code/configs/qwen3_4b_export_sft_for_dpo.yaml

uv run llamafactory-cli train \
  code/configs/qwen3_4b_dpo_codex_direct_v2.yaml
```

数据检查、评测、Adapter 合并和故障处理的完整命令见
[`doc/DPO_FINE_TUNING.md`](doc/DPO_FINE_TUNING.md)。

## 代码如何组织

`code/scripts/` 是可直接执行的课程步骤；`code/src/job_finetune_course/` 是每个步骤调用的核心功能。

| 步骤 | 执行脚本 | 核心模块 |
| --- | --- | --- |
| 0. 环境检查 | `00_check_env.py` | 独立检查脚本 |
| 1. 数据检查 | `01_check_data.py` | `data_processing.py` |
| 2. 数据集切分 | `02_split_dataset.py` | `dataset_split.py` |
| 3. 转换训练格式 | `03_prepare_training_data.py` | `data_processing.py` |
| 4. LoRA SFT | `llamafactory-cli train` | `code/configs/qwen3_4b_sft.yaml` |
| 5. 模型评估 | `05_evaluate_model.py` | `model_evaluation.py` |
| 6. 合并模型 | `llamafactory-cli export` | `code/configs/qwen3_4b_export.yaml` |
| 7. API 部署 | `07_serve_model.py` | `model_deployment.py` |
| 8. 服务压测 | `08_benchmark_service.py` | `model_deployment.py` |

## 小白建议执行顺序

### 教学演示（Mac CPU/MPS，用 1.7B 模型）

```bash
# 0. 检查环境（不需要 GPU）
uv run python code/scripts/00_check_env.py \
  --model-path ../../环境和物料/models/models/Qwen--Qwen3-1.7B/snapshots/master

# 1. 先用小样本理解数据结构
uv run python code/scripts/01_check_data.py --limit-per-task 100

# 2. 切分数据
uv run python code/scripts/02_split_dataset.py \
  --output-dir code/data/datasets/candidates/gemini-v2-split-demo

# 3. 格式转换
uv run python code/scripts/03_prepare_training_data.py \
  --output-dir code/outputs/demo_llamafactory_data

# 4. 小样本 SFT 训练（100 条 x 1 epoch，约 2 分钟）
uv run llamafactory-cli train code/configs/qwen3_1.7b_sft_demo.yaml

# 5. 评估
uv run python code/scripts/05_evaluate_model.py \
  --model-path ../../环境和物料/models/models/Qwen--Qwen3-1.7B/snapshots/master \
  --adapter-path code/outputs/runs/sft-qwen3-1.7b-demo \
  --dataset code/data/datasets/llamafactory/recruitment_sft_diagnostic.jsonl \
  --limit 20
```

### 正式训练（GPU 服务器，用 4B 或 8B 模型）

```bash
# 0. 检查 GPU、Python、依赖和基座模型
uv run python code/scripts/00_check_env.py --require-cuda

# 1. 先用小样本理解数据结构
uv run python code/scripts/01_check_data.py --limit-per-task 100

# 2. tasks_fixed 没有改变时，可以跳过此步，直接复用 split-v2
uv run python code/scripts/02_split_dataset.py \
  --output-dir code/data/datasets/candidates/gemini-v2-split-v2

# 3. 演示格式转换，输出到 outputs，不覆盖正式数据
uv run python code/scripts/03_prepare_training_data.py \
  --output-dir code/outputs/demo_llamafactory_data

# 4. 使用 LLaMA-Factory 原生命令执行 SFT（4B）
uv run llamafactory-cli train code/configs/qwen3_4b_sft.yaml

# 5. 不加载模型的评估链路自测
uv run python code/scripts/05_evaluate_model.py --gold-as-prediction --limit 20

# 6. 使用 LLaMA-Factory 合并 LoRA
uv run llamafactory-cli export code/configs/qwen3_4b_export.yaml

# 7. 先展示 API 启动命令，真正部署时删除 --dry-run
uv run python code/scripts/07_serve_model.py --dry-run

# 8. API 启动后测试并发、延迟、吞吐和稳定性
uv run python code/scripts/08_benchmark_service.py --requests 100 --concurrency 8
```

## 微调完成后加载模型并评测业务指标

评测入口是 `code/scripts/05_evaluate_model.py`，核心实现在
`code/src/job_finetune_course/model_evaluation.py`。脚本既可评测“基座模型 + LoRA Adapter”，
也可直接评测 `model-path` 下的基座模型或已合并模型。LoRA 加载关系如下：

```text
ShareGPT JSONL 格式的评测集
                    ↓
Qwen3-4B 基座模型 + 微调生成的 LoRA Adapter
                    ↓
        逐条生成 JSON 业务结果
                    ↓
 predictions.jsonl + metrics.json
```

### 1. 确认模型和数据路径

以 `code/configs/qwen3_4b_sft.yaml` 的默认训练配置为例：

| 内容 | 默认路径 | 要求 |
| --- | --- | --- |
| 基座模型 | `model_files/Qwen3-4B` | 应包含模型权重、Tokenizer 和 `config.json` |
| LoRA Adapter | `code/outputs/runs/sft-qwen3-4b-lf` | 应包含 `adapter_config.json` 和 Adapter 权重 |
| 调试集 | `code/data/datasets/llamafactory/recruitment_sft_diagnostic.jsonl` | 用于查错和快速对比 |
| 冻结测试集 | `code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl` | 用于最终业务验收，不应反向用于调参 |

评测器同时支持两种 ShareGPT 数据结构：SFT 数据取 `conversations` 最后的
assistant 消息作为标准答案；DPO 数据取顶层 `chosen` 作为标准答案，
并忽略只用于偏好训练的 `rejected`。

如果训练时修改了 YAML 中的 `model_name_or_path` 或 `output_dir`，
评测 LoRA 时必须将对应的实际路径传给 `--model-path` 和 `--adapter-path`。
如果 `--model-path` 已经是合并后的完整模型，则只传 `--model-path`，不要传
`--adapter-path`。

### 2. 先做评测链路自检

下面的命令不加载模型，而是把标准答案当作预测结果，用于检查数据读取、
JSON 解析、指标计算和文件写入链路。此时四项比率应均为 `1.0`。

```bash
uv run python code/scripts/05_evaluate_model.py \
  --dataset code/data/datasets/llamafactory/recruitment_sft_diagnostic.jsonl \
  --output-dir code/outputs/evaluation/pipeline-check \
  --gold-as-prediction \
  --limit 20
```

### 3. 加载微调模型做小样本冒烟评测

正式全量评测前，先生成 20 条数据，确认基座模型、Adapter、显存和输出格式正常：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path model_files/Qwen3-4B \
  --adapter-path code/outputs/runs/sft-qwen3-4b-lf \
  --dataset code/data/datasets/llamafactory/recruitment_sft_diagnostic.jsonl \
  --output-dir code/outputs/evaluation/sft-smoke \
  --limit 20 \
  --batch-size 4 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

`--limit 0` 表示不限制样本数，也是脚本默认值。`--batch-size` 表示每次同时
生成的样本数，默认为 `1`；增大它可以提高 GPU 利用率和评测吞吐，但会占用更多显存。
建议从 `4` 开始，再根据显存逐步调大。NVIDIA GPU 优先使用 `bf16`；
硬件不支持 BF16 时可改为 `--torch-dtype fp16`。

### 4. 在冻结测试集上做全量业务评测

冒烟评测通过后，去掉 `--limit` 并切换到 `frozen_test`：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path model_files/Qwen3-4B \
  --adapter-path code/outputs/runs/sft-qwen3-4b-lf \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/sft-frozen-test \
  --batch-size 4 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

执行时，脚本会先通过 Transformers 加载 `--model-path` 指向的基座模型，
再通过 PEFT 挂载 `--adapter-path` 中的 LoRA 权重。生成使用固定的贪心解码
`do_sample=False`，便于不同版本模型重复对比。
如果未指定 `--adapter-path`，则跳过 PEFT 挂载，直接使用 `--model-path` 生成预测。

例如，直接评测 `llamafactory-cli export` 导出的合并模型：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path code/outputs/merged_qwen3_4b_recruitment \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/merged-sft-frozen-test \
  --batch-size 4 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

### 5. 查看评测产物和指标

上述正式评测会生成：

```text
code/outputs/evaluation/sft-frozen-test/
├── metrics.json       # 汇总业务指标
└── predictions.jsonl # 每条样本的输入、标准答案、模型输出和解析结果
```

`metrics.json` 中的指标含义：

| 指标 | 含义 | 观察重点 |
| --- | --- | --- |
| `json_parse_rate` | 模型输出能成功解析出 JSON 对象的样本比例 | 反映输出格式稳定性 |
| `schema_exact_rate` | 预测 JSON 与标准答案的字段集完全一致的样本比例 | 反映少字段或多字段问题 |
| `field_exact_rate` | 标准答案中各字段值精确匹配的微平均比例 | 作为整体字段级业务准确率 |
| `object_exact_rate` | 整个预测 JSON 与标准答案完全一致的样本比例 | 最严格的单样本全对率 |

`metrics.json` 还包含 `by_task` 和 `classification`。其中 `classification.by_task`
覆盖全部八个分类任务：岗位名称、taxonomy 叶子到父级、taxonomy 二级到子级、用工类型、
工作场所、薪资结算、工作时间和员工福利分类。每个任务输出 `accuracy`、分类字段准确率、
macro precision/recall/F1、逐类别指标和稀疏混淆矩阵；多标签的
`taxonomy_level2_to_children` 额外输出集合 exact accuracy 和 micro/macro P/R/F1。

评测目录还会生成 `classification_metrics.md`，用于直接查看全部分类任务汇总表。
分类准确率只比较真正的业务标签字段，不把 `entity`、`position`、`rawWorkPlace`
等输入回显字段算作分类标签。

建议将不同模型版本写入不同的 `--output-dir`，不要相互覆盖。比较时先看
`json_parse_rate`，再以 `field_exact_rate` 作为主业务指标，同时用
`object_exact_rate` 衡量整条结果的严格正确率。如果指标下降，直接查看
`predictions.jsonl` 中的 `gold_json` 和 `prediction_json` 做错误定位。

> 注：执行 `uv run llamafactory-cli export <config>` 后的合并模型已经包含 LoRA 权重，
> 评测合并模型时只传 `--model-path`，不要重复挂载 Adapter。

## 重要说明

- 已有 `gemini-v2-split-v2` 和 LLaMA-Factory 数据可直接用于训练，不需要每次重新生成。
- 所有 SFT/DPO 微调均使用 `uv run llamafactory-cli train <config>`。
- 本次代码精简不改写现有 `code/data/**`、`code/outputs/**` 和 `model_files/**`。
- 被替换的复杂实现保存在 `archive/project_simplification_2026-07-10/beginner_refactor_2026-07-10/`。
