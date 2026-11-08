# Qwen3-8B 服务器微调操作指南

## 1. 适用范围

本指南用于在 Linux + NVIDIA GPU 服务器上完成 Qwen3-8B 的：

1. 基座模型评测；
2. LLaMA-Factory LoRA SFT；
3. SFT Adapter 评测与合并；
4. 从 SFT 合并模型开始的 LoRA DPO；
5. DPO Adapter 评测、合并和服务冒烟。

本地环境没有目标 NVIDIA GPU，因此本文中的路径与 YAML 已做静态一致性检查，但训练显存、速度、CUDA 兼容性和最终指标必须在服务器上验证。

## 2. 8B 配置清单

| 阶段 | 配置 | 输出 |
| --- | --- | --- |
| SFT | `code/configs/qwen3_8b_sft.yaml` | `code/outputs/runs/sft-qwen3-8b-lf` |
| 合并 SFT | `code/configs/qwen3_8b_export_sft_for_dpo.yaml` | `code/outputs/merged_qwen3_8b_recruitment_sft` |
| DPO | `code/configs/qwen3_8b_dpo_codex_direct_v2.yaml` | `code/outputs/runs/dpo-qwen3-8b-codex-direct-v2` |
| 合并 DPO | `code/configs/qwen3_8b_dpo_codex_direct_v2_export.yaml` | `code/outputs/merged_qwen3_8b_recruitment_dpo_codex_direct_v2` |
| API | `code/configs/qwen3_8b_api.yaml` | OpenAI 兼容服务 |

配置是保守的单卡 BF16 LoRA 起点：`cutoff_len=2048`、单卡 batch size 1、梯度累积 16、梯度检查点。它不是对任意显卡的显存承诺。

### 2.1 本项目统一关闭思考模式

本项目是招聘领域实体抽取和分类任务，训练目标是直接输出一个 JSON 对象。SFT 答案、
DPO chosen/rejected 都不包含思维链，因此 Qwen3-8B 的 SFT、DPO、评测和部署统一使用
**非思考模式**：

```yaml
template: qwen3
enable_thinking: false
```

上述设置已经写入：

- `code/configs/qwen3_8b_sft.yaml`
- `code/configs/qwen3_8b_dpo_codex_direct_v2.yaml`
- `code/configs/qwen3_8b_api.yaml`

不要把 `template` 换成普通模板，也不要在同一次实验中混用 `enable_thinking: true` 和
`false`。LLaMA-Factory 会在非思考模式下把空的 `<think></think>` 边界放入输入侧，
但不把思维链作为训练目标；模型学习的回答仍是最终 JSON。空思考边界是预期行为，
不表示模型正在生成思维过程。

关闭思考模式的原因：

- 语料没有可监督的思维链，强行开启不会增加有效标签信息；
- 招聘抽取和分类需要稳定、低延迟的结构化结果；
- 减少思考文本、解释或 Markdown 混入 JSON 的风险；
- 让训练、离线评测和线上 API 使用同一种 chat template 口径。

只有未来单独建设了高质量 `<think>...</think>` 推理语料，并且业务确实需要复杂推理时，
才应创建独立实验开启思考模式；该实验不能直接与本指南的非思考基线混合比较。

### 2.2 非思考模式静态检查

上服务器后可在训练前检查三份配置：

```bash
uv run python - <<'PY'
from pathlib import Path

import yaml

paths = [
    Path("code/configs/qwen3_8b_sft.yaml"),
    Path("code/configs/qwen3_8b_dpo_codex_direct_v2.yaml"),
    Path("code/configs/qwen3_8b_api.yaml"),
]
for path in paths:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["template"] == "qwen3", path
    assert config["enable_thinking"] is False, path
    print(path, "non-thinking OK")
PY
```

再检查 Qwen3-8B tokenizer 的实际渲染结果：

```bash
uv run python - <<'PY'
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "model_files/Qwen3-8B", local_files_only=True
)
messages = [
    {"role": "system", "content": "仅返回 JSON。"},
    {"role": "user", "content": "实体：兼职"},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
assert "<think>\n\n</think>" in prompt
print(prompt)
print("Qwen3-8B non-thinking template OK")
PY
```

训练后查看评测目录的 `predictions.jsonl`。模型新生成的 `prediction` 应直接是 JSON，
不应包含非空 `<think>...</think>`、解释文字或 Markdown。`json_parse_rate` 和
`schema_exact_rate` 是这项检查的主要守门指标。

## 3. 上机前准备

### 3.1 先完成 4B 门禁

建议在同一服务器、同一镜像上先完成 Qwen3-4B 的 20-step SFT 与 DPO 冒烟。4B 用于尽早发现：

- CUDA/PyTorch 不匹配；
- 模型或 tokenizer 文件不完整；
- 数据注册名、JSONL 格式或模板错误；
- checkpoint、评测和合并链路不可用。

4B 通过后仍需单独执行 8B 冒烟，因为模型权重和激活显存均不同。

### 3.2 将项目传到服务器

本地打包：

```bash
# 默认不包含 model_files
bash code/scripts/09_pack_for_server.sh

# 需要把 model_files/Qwen3-4B、Qwen3-8B 等模型一起打包时
bash code/scripts/09_pack_for_server.sh --include-models

# macOS
shasum -a 256 dist/recruitment-sft-*.tar.gz
# Linux 服务器收到文件后
sha256sum dist/recruitment-sft-*.tar.gz
```

带模型的包名包含 `-with-models`。执行前先用 `du -sh model_files` 和 `df -h .`
检查模型目录大小及剩余磁盘空间；压缩大模型可能耗时较长，并需要额外空间保存压缩包。

将压缩包传到服务器后解压。若使用默认包，模型权重不在压缩包内，应单独放置：

```text
model_files/Qwen3-4B/   # 低成本链路验证，可选但推荐
model_files/Qwen3-8B/   # 正式业务模型，必需
```

不要在文档或脚本中写模型仓库令牌。通过服务器密钥管理或已授权的离线传输准备模型。

### 3.3 记录服务器基线

在项目根目录执行并保存输出：

```bash
mkdir -p code/outputs/server_checks
nvidia-smi | tee code/outputs/server_checks/nvidia-smi.txt
python3 --version | tee code/outputs/server_checks/python-system.txt
df -h . | tee code/outputs/server_checks/disk.txt
```

合并 8B 模型会额外占用完整权重的磁盘空间；训练输出、checkpoint、评测预测和两份合并模型也需要空间。开始前应预留明显高于单份基座权重的容量。

## 4. 安装与环境检查

项目锁定 Python 3.11 和 LLaMA-Factory 0.9.4。服务器存在 `uv` 时执行：

```bash
uv sync --frozen
uv run python code/scripts/00_check_env.py \
  --model-path model_files/Qwen3-8B \
  --require-cuda | tee code/outputs/server_checks/qwen3-8b-env.json
```

再记录实际软件版本：

```bash
uv run python - <<'PY'
from importlib.metadata import version

import torch
import transformers
import peft

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("bf16_supported", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("llamafactory", version("llamafactory"))
PY
```

放行条件：环境检查返回 0，能识别目标 GPU，`config.json` 与 safetensors 存在，并且 BF16 支持。若 GPU 不支持 BF16，应先在复制的运行配置中关闭 `bf16`、启用 `fp16`，重新做 20-step 冒烟；不要直接修改仓库基准配置。

## 5. 数据预检

### 5.1 SFT 数据

```bash
uv run python code/scripts/01_check_data.py

wc -l \
  code/data/datasets/llamafactory/recruitment_sft_train.jsonl \
  code/data/datasets/llamafactory/recruitment_sft_dev.jsonl \
  code/data/datasets/llamafactory/recruitment_sft_diagnostic.jsonl \
  code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl
```

当前发布数据预期分别为 74326、4810、9497、4621 行。如果语料版本变化，应以该版本 manifest 为准，并同步更新记录，不能为了匹配本文数字删除样本。

### 5.2 DPO 数据与注册

```bash
uv run python - <<'PY'
import json
from pathlib import Path

data_dir = Path("code/data/datasets/llamafactory")
info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
expected = {
    "recruitment_dpo_codex_direct_v2_train": 900,
    "recruitment_dpo_codex_direct_v2_eval": 100,
}

for name, count in expected.items():
    spec = info[name]
    assert spec["formatting"] == "sharegpt"
    assert spec["ranking"] is True
    rows = [
        json.loads(line)
        for line in (data_dir / spec["file_name"]).open(encoding="utf-8")
        if line.strip()
    ]
    assert len(rows) == count, (name, len(rows), count)
    for index, row in enumerate(rows, start=1):
        assert row["conversations"], (name, index)
        assert row["chosen"]["from"] == "gpt", (name, index)
        assert row["rejected"]["from"] == "gpt", (name, index)
        assert row["chosen"]["value"] != row["rejected"]["value"], (name, index)
    print(name, len(rows), "OK")
PY
```

### 5.3 评测链路自检

此命令不加载模型，预期四项比率均为 `1.0`：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test_quick_30pct.jsonl \
  --output-dir code/outputs/evaluation/qwen3-8b-pipeline-check \
  --gold-as-prediction \
  --limit 20
```

## 6. 建立 Qwen3-8B 基座基线

先在 30% 固定快速集评测，确认能加载与生成：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path model_files/Qwen3-8B \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test_quick_30pct.jsonl \
  --output-dir code/outputs/evaluation/qwen3-8b-base-quick30 \
  --batch-size 2 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

显存稳定后可逐步增大评测 `--batch-size`，每次只改一个变量。正式对比前在完整冻结测试集建立 8B 基线：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path model_files/Qwen3-8B \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/qwen3-8b-base-frozen-test \
  --batch-size 1 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

保存这两次的 `metrics.json`、`classification_metrics.md` 和 `predictions.jsonl`。
其中 `classification_metrics.md` 汇总全部八个分类任务，逐类别指标和混淆矩阵位于
`metrics.json` 的 `classification.by_task`。没有 8B 自身基线，就无法证明 SFT 带来提升。

## 7. Qwen3-8B SFT

### 7.1 先做 20-step 冒烟

复制配置，不修改基准文件：

```bash
cp code/configs/qwen3_8b_sft.yaml \
  code/configs/qwen3_8b_sft_smoke.yaml
```

在副本中修改以下字段，避免 20-step 冒烟触发完整 dev 评测：

```yaml
do_eval: false
eval_strategy: "no"
load_best_model_at_end: false
max_steps: 20
save_steps: 20
save_total_limit: 1
output_dir: code/outputs/runs/sft-qwen3-8b-smoke20
```

启动并保留日志：

```bash
mkdir -p code/outputs/logs
set -o pipefail
uv run llamafactory-cli train \
  code/configs/qwen3_8b_sft_smoke.yaml \
  2>&1 | tee code/outputs/logs/qwen3-8b-sft-smoke20.log
```

冒烟放行条件：

- 训练完成 20 个 optimizer step 且退出码为 0；
- loss 为有限值，没有 NaN/Inf；
- 没有 CUDA OOM、NCCL、tokenizer 或数据解析错误；
- 输出目录存在 Adapter 权重和 `adapter_config.json`；
- 峰值显存仍有合理余量，或至少在训练全过程稳定。
- 冒烟配置仍保留 `template: qwen3` 和 `enable_thinking: false`。

### 7.2 正式 SFT

基准配置首轮使用 1 epoch。不要复用 smoke 输出目录：

```bash
set -o pipefail
uv run llamafactory-cli train \
  code/configs/qwen3_8b_sft.yaml \
  2>&1 | tee code/outputs/logs/qwen3-8b-sft.log
```

若中断，先确认最后一个完整 `checkpoint-*`。复制 YAML，加入：

```yaml
resume_from_checkpoint: code/outputs/runs/sft-qwen3-8b-lf/checkpoint-实际步数
```

再从副本恢复。不要设置 `overwrite_output_dir: true` 清空已有结果。

### 7.3 SFT 评测

先跑 20 条，再跑快速集，最后跑完整冻结集：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path model_files/Qwen3-8B \
  --adapter-path code/outputs/runs/sft-qwen3-8b-lf \
  --dataset code/data/datasets/llamafactory/recruitment_sft_diagnostic.jsonl \
  --output-dir code/outputs/evaluation/qwen3-8b-sft-smoke20 \
  --limit 20 \
  --batch-size 1 \
  --max-new-tokens 512 \
  --torch-dtype bf16

uv run python code/scripts/05_evaluate_model.py \
  --model-path model_files/Qwen3-8B \
  --adapter-path code/outputs/runs/sft-qwen3-8b-lf \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test_quick_30pct.jsonl \
  --output-dir code/outputs/evaluation/qwen3-8b-sft-quick30 \
  --batch-size 128 \
  --max-new-tokens 512 \
  --torch-dtype bf16

uv run python code/scripts/05_evaluate_model.py \
  --model-path model_files/Qwen3-8B \
  --adapter-path code/outputs/runs/sft-qwen3-8b-lf \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/qwen3-8b-sft-frozen-test \
  --batch-size 1 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

只有完整冻结集相对 8B 基座满足预先约定的提升和守门阈值，才进入 DPO。

### 7.4 合并 SFT，作为 DPO 起点

```bash
uv run llamafactory-cli export \
  code/configs/qwen3_8b_export_sft_for_dpo.yaml

test -f code/outputs/merged_qwen3_8b_recruitment_sft/config.json
```

合并后建议再抽取 20 条评测，确认合并模型与“基座 + Adapter”输出口径一致。合并模型评测时只传 `--model-path`。

## 8. Qwen3-8B DPO

### 8.1 起点检查

确认 DPO YAML 的 `model_name_or_path` 是刚生成的 8B SFT 合并模型，而不是原始 8B 基座：

```bash
test -f code/outputs/merged_qwen3_8b_recruitment_sft/config.json
test -f code/data/datasets/llamafactory/recruitment_dpo_codex_direct_v2_train.jsonl
test -f code/data/datasets/llamafactory/recruitment_dpo_codex_direct_v2_eval.jsonl
```

### 8.2 先做 20-step DPO 冒烟

```bash
cp code/configs/qwen3_8b_dpo_codex_direct_v2.yaml \
  code/configs/qwen3_8b_dpo_codex_direct_v2_smoke.yaml
```

在副本中修改以下字段，避免冒烟阶段执行完整偏好评测：

```yaml
do_eval: false
eval_strategy: "no"
load_best_model_at_end: false
max_steps: 20
save_steps: 20
save_total_limit: 1
output_dir: code/outputs/runs/dpo-qwen3-8b-smoke20
```

执行：

```bash
set -o pipefail
uv run llamafactory-cli train \
  code/configs/qwen3_8b_dpo_codex_direct_v2_smoke.yaml \
  2>&1 | tee code/outputs/logs/qwen3-8b-dpo-smoke20.log
```

DPO 会同时处理 chosen/rejected，通常比 SFT 更容易出现显存压力。SFT 能运行不能作为 DPO 一定能运行的证明。
DPO 的配置副本也必须保留 `enable_thinking: false`，不能只在 SFT 阶段关闭。

### 8.3 正式 DPO

```bash
set -o pipefail
uv run llamafactory-cli train \
  code/configs/qwen3_8b_dpo_codex_direct_v2.yaml \
  2>&1 | tee code/outputs/logs/qwen3-8b-dpo-codex-direct-v2.log
```

训练内的 DPO eval loss 用于选择 checkpoint，不代替冻结业务评测。

### 8.4 DPO 业务评测

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path code/outputs/merged_qwen3_8b_recruitment_sft \
  --adapter-path code/outputs/runs/dpo-qwen3-8b-codex-direct-v2 \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test_quick_30pct.jsonl \
  --output-dir code/outputs/evaluation/qwen3-8b-dpo-quick30 \
  --batch-size 128 \
  --max-new-tokens 512 \
  --torch-dtype bf16

uv run python code/scripts/05_evaluate_model.py \
  --model-path code/outputs/merged_qwen3_8b_recruitment_sft \
  --adapter-path code/outputs/runs/dpo-qwen3-8b-codex-direct-v2 \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/qwen3-8b-dpo-frozen-test \
  --batch-size 1 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

对比顺序必须是 `8B Base → 8B SFT → 8B DPO`。同时按任务与目标错误检查，防止总指标提升但某类业务明显退化。

### 8.5 合并最终 DPO 模型

仅在 DPO 评测放行后执行：

```bash
uv run llamafactory-cli export \
  code/configs/qwen3_8b_dpo_codex_direct_v2_export.yaml

test -f code/outputs/merged_qwen3_8b_recruitment_dpo_codex_direct_v2/config.json
```

对合并模型再跑 20 条和完整冻结集，确认合并前后无异常。评测合并模型时不要再挂载 Adapter。

## 9. 显存不足时的处理顺序

出现 CUDA OOM 时一次只改一项，并为新实验使用独立配置和输出目录：

1. 保持 `per_device_train_batch_size: 1` 和梯度检查点。
2. 关闭同机其他占用 GPU 的进程后重试。
3. 将 `cutoff_len` 从 2048 降至 1536 或 1024；先统计截断影响，再重新训练和评测。
4. 复制配置后改用 4-bit QLoRA，例如加入 `quantization_bit: 4` 和 `quantization_method: bitsandbytes`；先确认服务器 CUDA/bitsandbytes 兼容。
5. 仍无法运行时，换更大显存 GPU，或使用经过验证的 ZeRO-3/FSDP 多卡方案。

注意：普通数据并行会在每张卡复制模型，增加 GPU 数量不必然解决单卡 OOM。量化、截断或分布式策略改变后，必须重新建立性能与吞吐记录。QLoRA Adapter 合并时仍应以未量化的原始 8B 基座为合并底座。

## 10. 服务冒烟与压测

启动前先展示命令：

```bash
uv run python code/scripts/07_serve_model.py \
  --config code/configs/qwen3_8b_api.yaml \
  --model-name qwen3-8b-recruitment \
  --host 0.0.0.0 \
  --port 8000 \
  --dry-run
```

确认后删除 `--dry-run`。生产环境通过 `API_KEY` 配置认证，不要把密钥提交到仓库。
`qwen3_8b_api.yaml` 已设置 `enable_thinking: false`。若改用 vLLM、SGLang 或其他服务框架，
需要使用对应框架的非思考 chat template 或请求参数，不能假设训练配置会自动传递到新框架。

另一个终端执行：

```bash
uv run python code/scripts/08_benchmark_service.py \
  --model qwen3-8b-recruitment \
  --requests 100 \
  --concurrency 8 \
  --output code/outputs/qwen3-8b-service-benchmark.json
```

压测并发应从 1、2、4、8 逐级提高，记录成功率、RPS、输出 token/s 和 p95/p99 延迟。离线指标通过但服务稳定性不通过时，模型仍不能发布。

## 11. 最终验收清单

- [ ] 服务器、依赖、模型版本和数据 manifest 已记录。
- [ ] 4B 环境与链路验证通过，或有书面理由说明跳过。
- [ ] 8B 基座完整冻结集基线已保存。
- [ ] 8B SFT 20-step 冒烟、正式训练、Adapter 加载和合并通过。
- [ ] SFT、DPO、评测和 API 均使用 `template: qwen3`、`enable_thinking: false`。
- [ ] 8B SFT 相对 8B 基座达到预设业务阈值。
- [ ] 8B DPO 从 8B SFT 合并模型开始，而不是从原始基座开始。
- [ ] 8B DPO 冒烟、正式训练、Adapter 加载和合并通过。
- [ ] 8B DPO 的目标错误改善且关键 SFT 能力无不可接受回退。
- [ ] 最终合并模型在完整冻结集复评通过。
- [ ] API 冒烟、认证和分级并发压测通过。
- [ ] 配置、日志、checkpoint、metrics、predictions 和模型 hash 已归档。
