"""模型推理与最小业务指标评估。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

from .data_processing import read_jsonl, write_jsonl
from .project_config import TASK_NAMES

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


# 分类准确率只比较业务分类字段，不把 entity、position、rawWorkPlace 等输入回显字段算入标签。
CLASSIFICATION_TASK_FIELDS: dict[str, tuple[str, ...]] = {
    "job_name_classification": ("level1", "level2", "level3"),
    "taxonomy_leaf_to_parent": ("level1", "level2", "level3"),
    "taxonomy_level2_to_children": ("level1", "level2", "level3Children"),
    "work_type_classification": ("category",),
    "workplace_classification": ("level1", "level2"),
    "settlement_classification": ("category",),
    "work_time_classification": ("fullCategory", "level1", "level2"),
    "benefit_classification": ("standardBenefit",),
}

TASK_PROMPT_MARKERS: tuple[tuple[str, str | None], ...] = (
    ("员工福利标准化专家", "benefit_classification"),
    ("实体抽取标注专家", "entity_extraction"),
    ("岗位名称分类专家", "job_name_classification"),
    ("工作酬劳结算方式分类专家", "settlement_classification"),
    ("用工类型分类专家", "work_type_classification"),
    ("职位分类树关系专家", None),
    ("工作时间分类专家", "work_time_classification"),
    ("工作场所分类专家", "workplace_classification"),
)

INVALID_JSON_LABEL = "<INVALID_JSON>"
MISSING_LABEL = "<MISSING>"


def parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中找到第一个 JSON 对象。"""

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def infer_task_name(
    gold: dict[str, Any] | None, prompt_text: str = "", raw_task: Any = None
) -> str:
    """从显式 task、系统提示词或输出 schema 识别九个业务任务。"""

    if raw_task in TASK_NAMES:
        return str(raw_task)
    for marker, task in TASK_PROMPT_MARKERS:
        if marker not in prompt_text:
            continue
        if task is not None:
            return task
        if isinstance(gold, dict) and "level3Children" in gold:
            return "taxonomy_level2_to_children"
        return "taxonomy_leaf_to_parent"
    if not isinstance(gold, dict):
        return "unparsed_gold"
    keys = set(gold)
    exact_schema_tasks = {
        frozenset(
            {
                "benefits",
                "company",
                "description",
                "education",
                "employmentType",
                "location",
                "position",
                "preferredQualifications",
                "requirements",
                "responsibilities",
                "salary",
                "salaryPayment",
                "schedule",
                "workExperience",
                "workMode",
                "workPlace",
            }
        ): "entity_extraction",
        frozenset({"position", "level1", "level2", "level3"}): (
            "job_name_classification"
        ),
        frozenset({"level1", "level2", "level3Children"}): (
            "taxonomy_level2_to_children"
        ),
        frozenset({"rawWorkPlace", "level1", "level2"}): ("workplace_classification"),
        frozenset({"entity", "fullCategory", "level1", "level2"}): (
            "work_time_classification"
        ),
        frozenset({"entity", "standardBenefit"}): "benefit_classification",
    }
    task = exact_schema_tasks.get(frozenset(keys))
    if task:
        return task
    # taxonomy_leaf_to_parent 与 job_name_classification 的子 schema 相同，缺少提示词时无法可靠区分。
    if keys == {"level1", "level2", "level3"}:
        return "taxonomy_leaf_to_parent"
    if keys == {"entity", "category"}:
        return "generic_category_classification"
    return "unknown"


def load_evaluation_rows(dataset_path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    """读取 ShareGPT SFT/DPO 数据，拆分提示词和标准答案。"""

    rows = []
    for index, row in enumerate(read_jsonl(dataset_path), start=1):
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"无效 conversations：{dataset_path}:{index}")
        messages = [
            {"role": ROLE_MAP[item["from"]], "content": str(item.get("value", ""))}
            for item in conversations
        ]
        chosen = row.get("chosen")
        if isinstance(chosen, dict) and "value" in chosen:
            # DPO 格式：conversations 是完整提示词，chosen 是标准答案。
            prompt_messages = messages
            gold_text = str(chosen.get("value", ""))
        elif messages[-1]["role"] == "assistant":
            # SFT 格式：最后一条 assistant 消息是标准答案。
            prompt_messages = messages[:-1]
            gold_text = messages[-1]["content"]
        else:
            raise ValueError(
                f"样本缺少 assistant 或 chosen 标准答案：{dataset_path}:{index}"
            )
        gold_json = parse_json_object(gold_text)
        rows.append(
            {
                "sample_id": f"{dataset_path.stem}:{index}",
                "prompt_messages": prompt_messages,
                "gold": gold_text,
                "gold_json": gold_json,
                "task": infer_task_name(
                    gold_json,
                    "\n".join(message["content"] for message in prompt_messages),
                    row.get("task"),
                ),
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def load_model(
    model_path: Path, adapter_path: Path | None, torch_dtype: str
) -> tuple[Any, Any, str]:
    """加载 Qwen3 底座模型，并可选挂载 LoRA adapter。"""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float16
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device).eval()
    return model, tokenizer, device


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """将一组对话消息转换为模型推理文本。"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def generate_batch(
    model: Any,
    tokenizer: Any,
    message_batches: list[list[dict[str, str]]],
    max_new_tokens: int,
) -> list[str]:
    """对多条消息做 padding 后一次生成，并返回对应的预测文本。"""

    import torch

    if not message_batches:
        return []
    texts = [_render_prompt(tokenizer, messages) for messages in message_batches]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, input_length:]
    return [
        text.strip()
        for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    ]


def generate(
    model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int
) -> str:
    """用固定参数生成一条结果，保留单条调用接口。"""

    return generate_batch(model, tokenizer, [messages], max_new_tokens)[0]


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(inner) for key, inner in sorted(value.items())
        }
    return value


def _value_equal(value_a: Any, value_b: Any, *, unordered: bool = False) -> bool:
    if unordered and isinstance(value_a, list) and isinstance(value_b, list):
        normalized_a = {
            json.dumps(_normalize_value(item), ensure_ascii=False, sort_keys=True)
            for item in value_a
        }
        normalized_b = {
            json.dumps(_normalize_value(item), ensure_ascii=False, sort_keys=True)
            for item in value_b
        }
        return normalized_a == normalized_b
    return _normalize_value(value_a) == _normalize_value(value_b)


def _calculate_basic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """计算一组样本的 JSON、schema、字段和对象精确匹配指标。"""

    parsed = schema_exact = object_exact = correct_fields = total_fields = 0
    for row in rows:
        gold = row["gold_json"]
        prediction = row["prediction_json"]
        if prediction is not None:
            parsed += 1
        if not isinstance(gold, dict) or not isinstance(prediction, dict):
            continue
        schema_exact += set(gold) == set(prediction)
        object_exact += gold == prediction
        total_fields += len(gold)
        correct_fields += sum(
            prediction.get(key) == value for key, value in gold.items()
        )
    total = len(rows)
    return {
        "total": total,
        "json_parse_rate": _safe_rate(parsed, total),
        "schema_exact_rate": _safe_rate(schema_exact, total),
        "field_exact_rate": _safe_rate(correct_fields, total_fields),
        "object_exact_rate": _safe_rate(object_exact, total),
    }


def _render_label(value: dict[str, Any] | None, fields: tuple[str, ...]) -> str:
    if not isinstance(value, dict):
        return INVALID_JSON_LABEL
    parts = []
    for field_name in fields:
        field_value = value.get(field_name, MISSING_LABEL)
        if isinstance(field_value, (dict, list)):
            rendered = json.dumps(
                _normalize_value(field_value), ensure_ascii=False, sort_keys=True
            )
        else:
            rendered = str(_normalize_value(field_value))
        parts.append(f"{field_name}={rendered}")
    return " | ".join(parts)


def _single_label_report(
    gold_labels: list[str], prediction_labels: list[str]
) -> dict[str, Any]:
    """生成单标签/层级路径分类的 macro 指标和稀疏混淆矩阵。"""

    gold_counts = Counter(gold_labels)
    prediction_counts = Counter(prediction_labels)
    pair_counts = Counter(zip(gold_labels, prediction_labels, strict=True))
    labels = sorted(set(gold_counts) | set(prediction_counts))
    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in labels:
        tp = pair_counts[(label, label)]
        fp = prediction_counts[label] - tp
        fn = gold_counts[label] - tp
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
        )
        per_class[label] = {
            "support": gold_counts[label],
            "predicted": prediction_counts[label],
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        precisions.append(precision or 0.0)
        recalls.append(recall or 0.0)
        f1_scores.append(f1)
    confusion: dict[str, dict[str, int]] = defaultdict(dict)
    for (gold_label, prediction_label), count in sorted(pair_counts.items()):
        confusion[gold_label][prediction_label] = count
    correct = sum(
        gold == prediction
        for gold, prediction in zip(gold_labels, prediction_labels, strict=True)
    )
    return {
        "correct": correct,
        "accuracy": _safe_rate(correct, len(gold_labels)),
        "macro_precision": _safe_rate(sum(precisions), len(precisions)),
        "macro_recall": _safe_rate(sum(recalls), len(recalls)),
        "macro_f1": _safe_rate(sum(f1_scores), len(f1_scores)),
        "per_class": per_class,
        "confusion_matrix": dict(confusion),
    }


def _multilabel_report(
    gold_values: list[Any], prediction_values: list[Any]
) -> dict[str, Any]:
    """生成 taxonomy children 多标签集合的 micro/macro 指标。"""

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    total_tp = total_fp = total_fn = exact = 0
    for gold_value, prediction_value in zip(
        gold_values, prediction_values, strict=True
    ):
        gold_set = (
            {str(_normalize_value(item)) for item in gold_value}
            if isinstance(gold_value, list)
            else set()
        )
        prediction_set = (
            {str(_normalize_value(item)) for item in prediction_value}
            if isinstance(prediction_value, list)
            else set()
        )
        exact += gold_set == prediction_set
        for label in gold_set | prediction_set:
            if label in gold_set and label in prediction_set:
                stats[label]["tp"] += 1
            elif label in prediction_set:
                stats[label]["fp"] += 1
            else:
                stats[label]["fn"] += 1
        total_tp += len(gold_set & prediction_set)
        total_fp += len(prediction_set - gold_set)
        total_fn += len(gold_set - prediction_set)
    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label, counts in sorted(stats.items()):
        precision = _safe_rate(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_rate(counts["tp"], counts["tp"] + counts["fn"])
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
        )
        per_class[label] = {
            "support": counts["tp"] + counts["fn"],
            "true_positive": counts["tp"],
            "false_positive": counts["fp"],
            "false_negative": counts["fn"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        precisions.append(precision or 0.0)
        recalls.append(recall or 0.0)
        f1_scores.append(f1)
    micro_precision = _safe_rate(total_tp, total_tp + total_fp)
    micro_recall = _safe_rate(total_tp, total_tp + total_fn)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision is not None
        and micro_recall is not None
        and micro_precision + micro_recall > 0
        else 0.0
    )
    return {
        "set_exact_accuracy": _safe_rate(exact, len(gold_values)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": _safe_rate(sum(precisions), len(precisions)),
        "macro_recall": _safe_rate(sum(recalls), len(recalls)),
        "macro_f1": _safe_rate(sum(f1_scores), len(f1_scores)),
        "per_class": per_class,
        "confusion_matrix": None,
    }


def _classification_task_metrics(
    task: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    fields = CLASSIFICATION_TASK_FIELDS[task]
    total = len(rows)
    valid_predictions = sum(
        isinstance(row.get("prediction_json"), dict) for row in rows
    )
    field_correct = Counter()
    exact_correct = 0
    for row in rows:
        gold = row.get("gold_json")
        prediction = row.get("prediction_json")
        field_matches = []
        for field_name in fields:
            unordered = (
                task == "taxonomy_level2_to_children" and field_name == "level3Children"
            )
            matched = (
                isinstance(gold, dict)
                and isinstance(prediction, dict)
                and field_name in gold
                and field_name in prediction
                and _value_equal(
                    gold[field_name], prediction[field_name], unordered=unordered
                )
            )
            field_matches.append(matched)
            field_correct[field_name] += matched
        exact_correct += all(field_matches)
    result: dict[str, Any] = {
        "total": total,
        "correct": exact_correct,
        "classification_fields": list(fields),
        "prediction_json_parse_rate": _safe_rate(valid_predictions, total),
        "accuracy": _safe_rate(exact_correct, total),
        "per_field_accuracy": {
            field_name: _safe_rate(field_correct[field_name], total)
            for field_name in fields
        },
    }
    if task == "taxonomy_level2_to_children":
        result.update(
            _multilabel_report(
                [
                    (
                        row.get("gold_json", {}).get("level3Children")
                        if isinstance(row.get("gold_json"), dict)
                        else None
                    )
                    for row in rows
                ],
                [
                    (
                        row.get("prediction_json", {}).get("level3Children")
                        if isinstance(row.get("prediction_json"), dict)
                        else None
                    )
                    for row in rows
                ],
            )
        )
        result["metric_type"] = "multi_label_with_parent_path"
    else:
        report = _single_label_report(
            [_render_label(row.get("gold_json"), fields) for row in rows],
            [_render_label(row.get("prediction_json"), fields) for row in rows],
        )
        # report accuracy 与分类字段完全匹配 accuracy 定义相同；保留统一的顶层 accuracy。
        result.update(
            {
                key: value
                for key, value in report.items()
                if key not in {"correct", "accuracy"}
            }
        )
        result["metric_type"] = (
            "hierarchical_single_label" if len(fields) > 1 else "single_label"
        )
    return result


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """计算总体、逐任务以及全部分类任务的业务指标。"""

    metrics = _calculate_basic_metrics(rows)
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_task[str(row.get("task") or "unknown")].append(row)
    metrics["by_task"] = {
        task: _calculate_basic_metrics(rows_by_task.get(task, []))
        for task in TASK_NAMES
    }
    classification_by_task = {
        task: _classification_task_metrics(task, rows_by_task.get(task, []))
        for task in CLASSIFICATION_TASK_FIELDS
    }
    classification_total = sum(
        item["total"] for item in classification_by_task.values()
    )
    classification_correct = sum(
        item["correct"] for item in classification_by_task.values()
    )
    task_accuracies = [
        item["accuracy"]
        for item in classification_by_task.values()
        if item["accuracy"] is not None
    ]
    metrics["classification"] = {
        "summary": {
            "expected_tasks": list(CLASSIFICATION_TASK_FIELDS),
            "tasks_with_samples": sum(
                item["total"] > 0 for item in classification_by_task.values()
            ),
            "total": classification_total,
            "correct": classification_correct,
            "accuracy": _safe_rate(classification_correct, classification_total),
            "macro_task_accuracy": _safe_rate(
                sum(task_accuracies), len(task_accuracies)
            ),
        },
        "by_task": classification_by_task,
    }
    return metrics


def render_classification_metrics(metrics: dict[str, Any]) -> str:
    """将分类任务核心指标渲染为便于人工查看的 Markdown 表格。"""

    def percent(value: Any) -> str:
        return "NA" if value is None else f"{float(value) * 100:.2f}%"

    classification = metrics["classification"]
    summary = classification["summary"]
    lines = [
        "# 分类任务评测汇总",
        "",
        f"- 分类样本数：{summary['total']}",
        f"- 分类总体准确率：{percent(summary['accuracy'])}",
        f"- 任务宏平均准确率：{percent(summary['macro_task_accuracy'])}",
        f"- 有样本任务数：{summary['tasks_with_samples']} / {len(summary['expected_tasks'])}",
        "",
        "| 任务 | 样本数 | Accuracy | Macro Precision | Macro Recall | Macro F1 | JSON 解析率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, task_metrics in classification["by_task"].items():
        lines.append(
            "| {task} | {total} | {accuracy} | {precision} | {recall} | {f1} | {parse_rate} |".format(
                task=task,
                total=task_metrics["total"],
                accuracy=percent(task_metrics["accuracy"]),
                precision=percent(task_metrics.get("macro_precision")),
                recall=percent(task_metrics.get("macro_recall")),
                f1=percent(task_metrics.get("macro_f1")),
                parse_rate=percent(task_metrics["prediction_json_parse_rate"]),
            )
        )
    lines.extend(["", "## 各任务字段准确率", ""])
    for task, task_metrics in classification["by_task"].items():
        field_text = ", ".join(
            f"`{field}`={percent(value)}"
            for field, value in task_metrics["per_field_accuracy"].items()
        )
        lines.append(f"- **{task}**：{field_text or 'NA'}")
    lines.append("")
    return "\n".join(lines)


def evaluate_model(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_path: Path | None = None,
    adapter_path: Path | None = None,
    limit: int = 0,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    torch_dtype: str = "bf16",
    gold_as_prediction: bool = False,
) -> dict[str, Any]:
    """执行预测、保存明细并返回指标。"""

    if batch_size < 1:
        raise ValueError("batch_size 必须大于等于 1")
    rows = load_evaluation_rows(dataset_path, limit=limit)
    model = tokenizer = None
    if not gold_as_prediction:
        if model_path is None:
            raise ValueError("真实推理必须提供 model_path")
        model, tokenizer, _ = load_model(model_path, adapter_path, torch_dtype)
    prediction_seconds = 0.0
    progress = tqdm(
        total=len(rows),
        desc="模型预测",
        unit="sample",
        dynamic_ncols=True,
    )
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_started = perf_counter()
            predictions = (
                [row["gold"] for row in batch_rows]
                if gold_as_prediction
                else generate_batch(
                    model,
                    tokenizer,
                    [row["prompt_messages"] for row in batch_rows],
                    max_new_tokens,
                )
            )
            prediction_seconds += perf_counter() - batch_started
            for row, prediction in zip(batch_rows, predictions, strict=True):
                row["prediction"] = prediction
                row["prediction_json"] = parse_json_object(prediction)
            progress.update(len(batch_rows))
    finally:
        progress.close()
    metrics = calculate_metrics(rows)
    metrics["prediction_performance"] = {
        "samples": len(rows),
        "batch_size": batch_size,
        "elapsed_seconds": round(prediction_seconds, 3),
        "samples_per_second": (
            round(len(rows) / prediction_seconds, 4) if prediction_seconds > 0 else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "classification_metrics.md").write_text(
        render_classification_metrics(metrics), encoding="utf-8"
    )
    return metrics
