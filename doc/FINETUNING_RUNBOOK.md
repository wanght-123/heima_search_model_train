# 招聘领域模型微调总运行手册

## 1. 文档目标与模型定位

本项目的微调统一使用 **LLaMA-Factory 0.9.4**，分为两个连续阶段：

1. **SFT（监督微调）**：让模型学习招聘领域任务、输出 schema、字段边界和分类体系。
2. **DPO（直接偏好优化）**：在合格 SFT 模型上，用 chosen/rejected 偏好对修正容易混淆的业务行为。

项目仓库当前已经验证和记录的是 **Qwen3-4B**。它是低成本验证模型，用于快速回答：

- 租赁 GPU 的 CUDA、驱动、Python 和依赖是否能正常工作；
- LLaMA-Factory 能否读取本项目数据并完成训练、保存、恢复和合并；
- 语料格式、标签与任务分布是否适合训练；
- 相同冻结测试集上的基础业务指标是否相对基座模型提升。

业务正式模型是 **Qwen3-8B**。4B 通过只说明环境与方案具备继续验证的条件，**不等于 8B 显存一定足够，也不等于 8B 业务效果已经通过**。8B 必须重新执行基座评测、SFT、DPO 和冻结测试集验收。

## 2. 文档入口

| 目标 | 文档 |
| --- | --- |
| 理解并执行 Qwen3-4B SFT | [`SFT_FULL_PIPELINE.md`](SFT_FULL_PIPELINE.md) |
| 执行 Qwen3-4B DPO | [`DPO_FINE_TUNING.md`](DPO_FINE_TUNING.md) |
| 在服务器执行 Qwen3-8B SFT + DPO | [`QWEN3_8B_FINE_TUNING.md`](QWEN3_8B_FINE_TUNING.md) |
| 准备、检查和发布微调语料 | [`CORPUS_PREPARATION.md`](CORPUS_PREPARATION.md) |
| 了解 DPO 偏好对设计细则 | [`DPO_DATA_DESIGN_PLAN.md`](DPO_DATA_DESIGN_PLAN.md) |

## 3. 正确的训练关系

```text
同一份冻结测试集
        │
        ├── Qwen3-4B 基座 ── SFT ── SFT 合并模型 ── DPO ── 4B 方案验收
        │                         （低成本环境、数据、指标验证）
        │
        └── Qwen3-8B 基座 ── SFT ── SFT 合并模型 ── DPO ── 8B 业务验收
                                  （正式业务模型，单独建立基线）
```

关键限制：

- DPO 必须从同尺寸、同版本、已通过验收的 SFT 合并模型开始。
- 4B Adapter 不能挂载到 8B，4B 合并模型也不能作为 8B DPO 起点。
- 基座、SFT、DPO 必须使用相同的 tokenizer、`template: qwen3` 和推理口径。
- 本项目语料输出是直接 JSON，不包含思维链；Qwen3-8B 的 SFT、DPO 和 API 配置均显式设置
  `enable_thinking: false`，评测代码也使用非思考模式。四个阶段必须保持一致。
- 冻结测试集只用于比较和最终验收，不能依据其答案返工当轮训练集。

## 4. 推荐执行顺序

### 4.1 本地可做的静态准备

本地无 NVIDIA 服务器时只做以下工作，不把它们表述为训练验证：

1. 检查 JSON/JSONL 可解析、schema 合法、chosen 与 rejected 不同。
2. 确认 `dataset_info.json` 注册名与 YAML 一致。
3. 确认模型路径、Adapter 路径和输出路径前后衔接。
4. 运行不加载模型的评测链路自检。
5. 将代码、配置、数据、文档和锁文件打包。

```bash
uv run python code/scripts/01_check_data.py
uv run python code/scripts/05_evaluate_model.py --gold-as-prediction --limit 20
bash code/scripts/09_pack_for_server.sh
```

默认服务器包不含模型。如确需把整个 `model_files/` 一并打包，使用
`bash code/scripts/09_pack_for_server.sh --include-models`，并先确认本地磁盘空间足够。

### 4.2 服务器上的 4B 低成本验证

按以下门禁逐级执行，前一步失败时不要直接开始 8B：

1. `00_check_env.py --require-cuda` 通过。
2. Qwen3-4B 基座在冻结测试集上完成基线评测。
3. 先复制配置并缩小为 20 个 optimizer step 的 SFT 冒烟任务。
4. 完整 SFT 结束，Adapter 可加载、可合并、指标相对 4B 基座提升。
5. 从 4B SFT 合并模型开始 DPO 冒烟与完整训练。
6. DPO 的目标错误改善，SFT 已获得的能力没有不可接受的回退。

4B 的详细命令分别见 SFT 和 DPO 手册。租赁按小时计费时，建议先完成 4B 冒烟再决定是否延长机器时长。

### 4.3 服务器上的 8B 正式训练

4B 门禁通过后，按 8B 专册执行。8B 需要自己的三组可比结果：

| 版本 | 用途 |
| --- | --- |
| Qwen3-8B 基座 | 8B 业务基线 |
| Qwen3-8B + SFT Adapter | 判断 SFT 是否学习到任务与 schema |
| Qwen3-8B SFT 合并模型 + DPO Adapter | 判断偏好优化收益与能力回退 |

不要用“8B 分数高于 4B”代替阶段增益分析。应分别计算：

```text
SFT 增益 = 8B-SFT 指标 - 8B-Base 指标
DPO 增益 = 8B-DPO 指标 - 8B-SFT 指标
```

## 5. 评测与放行标准

项目最小评测器输出：

| 指标 | 含义 | 放行用途 |
| --- | --- | --- |
| `json_parse_rate` | 输出能解析为 JSON 对象的比例 | 格式守门指标 |
| `schema_exact_rate` | 字段集合与标准答案完全一致的比例 | schema 守门指标 |
| `field_exact_rate` | 字段值精确匹配的微平均比例 | 主要业务指标 |
| `object_exact_rate` | 整个 JSON 完全一致的比例 | 严格整体指标 |

分类任务同时查看 `metrics.json` 的 `classification.by_task` 和同目录下的
`classification_metrics.md`。报告覆盖岗位名称、taxonomy 两类关系、用工类型、工作场所、
薪资结算、工作时间和员工福利共八个分类任务，并提供准确率、字段准确率、macro P/R/F1、
逐类别指标和混淆矩阵；taxonomy children 多标签任务使用集合与多标签 P/R/F1。

每次正式实验开始前，应在实验记录中预先写明阈值。若业务方尚未给出阈值，可采用以下决策原则，但不要事后为某次结果修改规则：

- SFT：主要业务指标相对同尺寸基座明确提升；格式和 schema 指标不得出现不可接受回退。
- DPO：预先指定的目标错误明显减少；同时检查九类任务，不能用总分掩盖单任务退化。
- 最终放行：完整冻结测试集通过，而不是只看 diagnostic 或 30% 快速集。
- 任何阶段都保存 `metrics.json`、`predictions.jsonl`、训练配置、日志、checkpoint 与模型版本信息。

若指标波动小于业务可识别差异，应视为“尚无充分证据”，而不是自动判定提升。

## 6. 服务器实验记录模板

每个实验目录至少记录以下内容：

```text
run_id:
date:
git_commit_or_package_hash:
gpu_name_and_count:
gpu_memory:
driver_and_cuda:
python_torch_transformers_llamafactory_versions:
base_model_path_and_hash:
dataset_manifest_and_counts:
config_file:
random_seed:
start_and_end_time:
peak_gpu_memory:
checkpoint_used:
evaluation_dataset:
metrics_path:
decision: pass / fail / retry
```

训练时不要直接修改仓库基准 YAML。复制成带 `run_id` 的配置后修改，并保证 `output_dir` 唯一。这样可以避免覆盖已有 checkpoint，也能复现实验。

## 7. 不能由本文档替代的验证

本次文档是在无目标 GPU 的本地环境中编写，只能完成路径、格式和配置的一致性检查。以下结论必须在服务器实测：

- BF16 是否被目标 GPU 与当前 PyTorch/CUDA 组合支持；
- 8B SFT、尤其 8B DPO 的峰值显存和吞吐；
- 服务器镜像、驱动、网络和模型文件是否完整；
- 中断恢复、合并模型加载、API 启动和并发压测是否成功；
- 4B 与 8B 在冻结测试集上的实际业务增益。

因此，操作顺序采用“静态检查 → 4B 冒烟 → 4B 完整链路 → 8B 冒烟 → 8B 完整链路”，而不是一次性假设正式训练必然成功。
