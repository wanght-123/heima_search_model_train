# Qwen3-4B Codex Direct v2 DPO 微调手册

本手册微调的是 **Qwen3-4B**，用于低成本验证 DPO 数据、租赁 GPU 环境和完整偏好训练链路。
业务正式模型是 Qwen3-8B，其独立操作见 [`QWEN3_8B_FINE_TUNING.md`](QWEN3_8B_FINE_TUNING.md)。
4B DPO 通过不表示 8B DPO 显存或业务指标自动通过。

本手册对应当前已发布的 1000 条 Codex Direct v2 DPO 数据。在项目根目录按下列步骤执行，
可完成数据检查、SFT 起点模型准备、DPO LoRA 训练、评测和模型合并。DPO 必须位于 SFT 之后，
不能用来替代尚未完成的领域能力学习。

## 1. 本次数据与配置

| 用途 | 文件或注册名 | 数量 |
| --- | --- | ---: |
| DPO 训练文件 | `code/data/datasets/llamafactory/recruitment_dpo_codex_direct_v2_train.jsonl` | 900 |
| DPO 评测文件 | `code/data/datasets/llamafactory/recruitment_dpo_codex_direct_v2_eval.jsonl` | 100 |
| 训练数据集名 | `recruitment_dpo_codex_direct_v2_train` | 900 |
| 评测数据集名 | `recruitment_dpo_codex_direct_v2_eval` | 100 |
| DPO 训练配置 | `code/configs/qwen3_4b_dpo_codex_direct_v2.yaml` | - |
| DPO Adapter 输出 | `code/outputs/runs/dpo-qwen3-4b-codex-direct-v2` | - |

`dataset_info.json` 已按 ShareGPT ranking 格式注册两个数据集。`conversations` 是提示词，`chosen` 是业务规则下的优选答案，`rejected` 是指定错误类型的困难负样本。

## 2. 环境和路径检查

项目统一使用 Python 3.11、`uv` 和 LLaMA-Factory 0.9.4。首次部署时执行：

```bash
cd /path/to/模型微调
uv sync
uv run python code/scripts/00_check_env.py --require-cuda
```

确认下列文件存在：

```bash
test -f model_files/Qwen3-4B/config.json
test -f code/outputs/runs/sft-qwen3-4b-lf-4090-fast/adapter_config.json
test -f code/data/datasets/llamafactory/recruitment_dpo_codex_direct_v2_train.jsonl
test -f code/data/datasets/llamafactory/recruitment_dpo_codex_direct_v2_eval.jsonl
test -f code/data/datasets/llamafactory/dataset_info.json
```

如果项目中已经产生了比 `sft-qwen3-4b-lf-4090-fast` 更新的完整 SFT Adapter，只需将 `code/configs/qwen3_4b_export_sft_for_dpo.yaml` 中的 `adapter_name_or_path` 改为实际路径。

## 3. 数据预检

先校验 JSONL 数量、ranking 结构和 `dataset_info.json` 注册关系：

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

for name, expected_count in expected.items():
    spec = info[name]
    assert spec["formatting"] == "sharegpt"
    assert spec["ranking"] is True
    path = data_dir / spec["file_name"]
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    assert len(rows) == expected_count, (name, len(rows))
    for row in rows:
        assert len(row["conversations"]) == 2
        assert row["conversations"][0]["from"] == "system"
        assert row["conversations"][1]["from"] == "human"
        assert row["chosen"]["from"] == "gpt"
        assert row["rejected"]["from"] == "gpt"
        assert row["chosen"]["value"] != row["rejected"]["value"]
    print(name, len(rows), "OK")
PY
```

预期输出：

```text
recruitment_dpo_codex_direct_v2_train 900 OK
recruitment_dpo_codex_direct_v2_eval 100 OK
```

## 4. 准备 DPO 起点模型

DPO 应从已经具备招聘任务能力的 SFT 模型开始。先将已完成的 SFT LoRA 合并到 Qwen3-4B 基座模型：

```bash
uv run llamafactory-cli export \
  code/configs/qwen3_4b_export_sft_for_dpo.yaml
```

合并后应存在：

```bash
test -f code/outputs/merged_qwen3_4b_recruitment/config.json
```

`code/configs/qwen3_4b_dpo_codex_direct_v2.yaml` 的 `model_name_or_path` 已指向该目录。不要把它改回未经 SFT 的 `model_files/Qwen3-4B`，否则 DPO 容易损害基础抽取和分类能力。

## 5. 执行 DPO LoRA 微调

确认第 3、4 节通过后，新租赁环境先复制配置并执行 20 个 optimizer step：

```bash
cp code/configs/qwen3_4b_dpo_codex_direct_v2.yaml \
  code/configs/qwen3_4b_dpo_codex_direct_v2_smoke.yaml
```

在副本中设置独立 `output_dir`，并修改：

```yaml
do_eval: false
eval_strategy: "no"
load_best_model_at_end: false
max_steps: 20
save_steps: 20
save_total_limit: 1
output_dir: code/outputs/runs/dpo-qwen3-4b-smoke20
```

```bash
uv run llamafactory-cli train \
  code/configs/qwen3_4b_dpo_codex_direct_v2_smoke.yaml
```

确认无 CUDA OOM、loss 非 NaN/Inf 且 Adapter 能保存后，再使用基准配置执行完整训练：

```bash
uv run llamafactory-cli train \
  code/configs/qwen3_4b_dpo_codex_direct_v2.yaml
```

默认参数适用于单卡、小规模 DPO 验证：

- batch size 1，梯度累积 16；
- learning rate `5e-6`；
- 1 epoch；
- DPO beta `0.1`；
- BF16 和梯度检查点；
- 每 20 个 optimizer step 评测和保存；
- 根据 `eval_loss` 恢复最优检查点。

使用项目本地 Qwen3-4B tokenizer 按完整 chat template 检查后，训练集 chosen/rejected 序列最长 1490 tokens，评测集最长 1346 tokens，均低于配置中的 `cutoff_len: 2048`。

训练输出位于：

```text
code/outputs/runs/dpo-qwen3-4b-codex-direct-v2/
```

如果中途停止，先查看该目录中的 `checkpoint-*`，不要直接删除旧输出或将 `overwrite_output_dir` 改为 `true`。

## 6. 训练后评测

### 6.1 DPO 训练内评测

LLaMA-Factory 在训练期间使用 `recruitment_dpo_codex_direct_v2_eval` 计算偏好评测 loss。这 100 条数据用于选择 checkpoint，不代替最终业务评测。

### 6.2 评测链路自检

在加载模型前，先确认业务评测器能正常读取冻结测试集：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/dpo-codex-direct-v2-pipeline-check \
  --gold-as-prediction \
  --limit 20
```

### 6.3 冒烟评测

DPO Adapter 训练完成后，使用“SFT 合并模型 + DPO Adapter”生成 20 条结果：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path code/outputs/merged_qwen3_4b_recruitment \
  --adapter-path code/outputs/runs/dpo-qwen3-4b-codex-direct-v2 \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/dpo-codex-direct-v2-smoke \
  --limit 20 \
  --batch-size 1 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

### 6.4 冻结测试集全量评测

冒烟通过后去掉 `--limit`：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path code/outputs/merged_qwen3_4b_recruitment \
  --adapter-path code/outputs/runs/dpo-qwen3-4b-codex-direct-v2 \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/dpo-codex-direct-v2-frozen-test \
  --batch-size 1 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

重点比较 DPO 前后的 `json_parse_rate`、`schema_exact_rate`、`field_exact_rate` 和 `object_exact_rate`。DPO 数据中不包含冻结测试集输入，评测结果可用于验证效果，不应回填到本轮训练数据。

## 7. 合并 DPO Adapter

评测确认 DPO 有收益后，再合并 Adapter：

```bash
uv run llamafactory-cli export \
  code/configs/qwen3_4b_dpo_codex_direct_v2_export.yaml
```

完整模型输出：

```text
code/outputs/merged_qwen3_4b_recruitment_dpo_codex_direct_v2/
```

合并后评测时只传 `--model-path`，不要再传 `--adapter-path`：

```bash
uv run python code/scripts/05_evaluate_model.py \
  --model-path code/outputs/merged_qwen3_4b_recruitment_dpo_codex_direct_v2 \
  --dataset code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl \
  --output-dir code/outputs/evaluation/dpo-codex-direct-v2-merged-frozen-test \
  --batch-size 1 \
  --max-new-tokens 512 \
  --torch-dtype bf16
```

## 8. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 找不到数据集 | 数据集名与 `dataset_info.json` 不一致 | 使用本手册给出的完整注册名 |
| 找不到 DPO 起点模型 | 未执行第 4 节的 SFT 合并 | 先执行 `qwen3_4b_export_sft_for_dpo.yaml` |
| CUDA OOM | DPO 同时计算 chosen/rejected，显存高于 SFT | 保持 batch size 1，必要时降低 `cutoff_len` |
| BF16 不支持 | GPU 架构较旧 | 将 `bf16: true` 改为 `bf16: false` 并加 `fp16: true` |
| 输出目录已存在 | 已有训练或中断 checkpoint | 使用 checkpoint 恢复，或换新 `output_dir`；不要盲目覆盖 |
| 评测时 Adapter 无法加载 | `--model-path` 不是 DPO 训练时的起点模型 | 使用 `code/outputs/merged_qwen3_4b_recruitment` |

## 9. 验收顺序

1. 数据预检通过，数量为 900/100。
2. SFT 合并模型存在且能加载。
3. LLaMA-Factory 能正确解析 DPO 配置与数据集注册。
4. DPO 训练产生 Adapter 和 eval loss。
5. 20 条冒烟评测能产生可解析 JSON。
6. 冻结测试集与 DPO 前 SFT 基线对比，目标错误改善且其他任务无明显退化。
7. 只在业务评测通过后合并和部署 DPO 模型。
