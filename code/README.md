# Demo 代码导读

读代码时请先看 `scripts/`，再跟着 import 进入 `src/job_finetune_course/`。

## scripts：课程入口

脚本只做三件事：接收命令行参数、调用核心模块、打印结果。

```text
00_check_env.py
01_check_data.py
02_split_dataset.py
03_prepare_training_data.py
04_train_model.py
05_evaluate_model.py
06_export_model.py
07_serve_model.py
08_benchmark_service.py
09_pack_for_server.sh
```

`09_pack_for_server.sh` 默认不包含模型；增加 `--include-models` 后会把整个
`model_files/` 加入压缩包，并生成带 `-with-models` 后缀的包名：

```bash
bash code/scripts/09_pack_for_server.sh --include-models
```

## src：核心功能

| 模块 | 只负责什么 |
| --- | --- |
| `project_config.py` | 项目根目录、九任务名、四个 split 名 |
| `data_processing.py` | JSONL 读写、样本检查、ShareGPT 转换 |
| `dataset_split.py` | 按 source_id 防泄漏切分 |
| `model_training.py` | 组装并执行 LLaMA-Factory train/export/api 命令 |
| `model_evaluation.py` | 模型加载、生成、JSON 解析和准确率 |
| `model_deployment.py` | API 启动、并发 HTTP 请求和性能统计 |

每个模块尽量保持单一职责，公开函数都有中文说明。

## DPO 入口

所有微调统一使用 LLaMA-Factory 原生训练命令，具体阶段由 YAML 的 `stage` 决定。Codex Direct v2 DPO 使用：

```bash
uv run llamafactory-cli train \
  code/configs/qwen3_4b_dpo_codex_direct_v2.yaml
```

详细顺序见项目文档 `doc/DPO_FINE_TUNING.md`。

## 4B 验证与 8B 正式训练

现有 `qwen3_4b_*` 配置用于快速验证服务器环境、数据和指标链路；正式业务模型使用
`qwen3_8b_*` 配置，并需要单独建立 8B 基座、SFT 和 DPO 三组评测结果。
总流程见 `doc/FINETUNING_RUNBOOK.md`，逐条服务器命令见
`doc/QWEN3_8B_FINE_TUNING.md`，语料规范见 `doc/CORPUS_PREPARATION.md`。
