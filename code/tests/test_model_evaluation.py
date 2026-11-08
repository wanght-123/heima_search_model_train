"""模型评估模块的最小测试。"""

from pathlib import Path

import pytest

from job_finetune_course import model_evaluation
from job_finetune_course.model_evaluation import (
    CLASSIFICATION_TASK_FIELDS,
    calculate_metrics,
    infer_task_name,
    parse_json_object,
)


def test_parse_json_and_metrics() -> None:
    assert parse_json_object('结果：```json\n{"category":"日结"}\n```') == {
        "category": "日结"
    }
    rows = [
        {"gold_json": {"category": "日结"}, "prediction_json": {"category": "日结"}},
        {"gold_json": {"category": "月结"}, "prediction_json": None},
    ]
    metrics = calculate_metrics(rows)
    assert metrics["json_parse_rate"] == 0.5
    assert metrics["object_exact_rate"] == 0.5


def test_load_evaluation_rows_supports_dpo_chosen(monkeypatch) -> None:
    dpo_row = {
        "conversations": [
            {"from": "system", "value": "return json"},
            {"from": "human", "value": "input text"},
        ],
        "chosen": {"from": "gpt", "value": '{"category": "A"}'},
        "rejected": {"from": "gpt", "value": '{"category": "B"}'},
    }
    monkeypatch.setattr(model_evaluation, "read_jsonl", lambda dataset_path: [dpo_row])

    rows = model_evaluation.load_evaluation_rows(Path("dpo_eval.jsonl"))

    assert rows[0]["prompt_messages"] == [
        {"role": "system", "content": "return json"},
        {"role": "user", "content": "input text"},
    ]
    assert rows[0]["gold_json"] == {"category": "A"}


@pytest.mark.parametrize(
    ("prompt", "gold", "expected"),
    [
        ("你是招聘领域员工福利标准化专家", {}, "benefit_classification"),
        ("你是招聘文本实体抽取标注专家", {}, "entity_extraction"),
        ("你是招聘领域岗位名称分类专家", {}, "job_name_classification"),
        ("你是工作酬劳结算方式分类专家", {}, "settlement_classification"),
        ("你是招聘领域用工类型分类专家", {}, "work_type_classification"),
        (
            "你是职位分类树关系专家",
            {"level3Children": []},
            "taxonomy_level2_to_children",
        ),
        (
            "你是职位分类树关系专家",
            {"level3": "Java"},
            "taxonomy_leaf_to_parent",
        ),
        ("你是招聘领域工作时间分类专家", {}, "work_time_classification"),
        ("你是招聘领域工作场所分类专家", {}, "workplace_classification"),
    ],
)
def test_infer_all_task_names(prompt, gold, expected) -> None:
    assert infer_task_name(gold, prompt) == expected


def test_classification_metrics_cover_all_classification_tasks() -> None:
    gold_by_task = {
        "job_name_classification": {
            "position": "Java工程师",
            "level1": "互联网/AI",
            "level2": "后端开发",
            "level3": "Java",
        },
        "taxonomy_leaf_to_parent": {
            "level1": "互联网/AI",
            "level2": "后端开发",
            "level3": "Java",
        },
        "taxonomy_level2_to_children": {
            "level1": "互联网/AI",
            "level2": "后端开发",
            "level3Children": ["Java", "Go"],
        },
        "work_type_classification": {"entity": "兼职", "category": "兼职工"},
        "workplace_classification": {
            "rawWorkPlace": "某仓库",
            "level1": "仓库",
            "level2": "仓库",
        },
        "settlement_classification": {"entity": "当天结", "category": "日结"},
        "work_time_classification": {
            "entity": "弹性打卡",
            "fullCategory": "弹性工时制-核心弹性",
            "level1": "弹性工时制",
            "level2": "核心弹性",
        },
        "benefit_classification": {
            "entity": "交通补贴",
            "standardBenefit": "交通补贴/班车",
        },
    }
    rows = []
    for task, gold in gold_by_task.items():
        prediction = dict(gold)
        if task == "taxonomy_level2_to_children":
            prediction["level3Children"] = list(reversed(gold["level3Children"]))
        rows.append({"task": task, "gold_json": gold, "prediction_json": prediction})

    metrics = calculate_metrics(rows)

    assert list(metrics["classification"]["by_task"]) == list(
        CLASSIFICATION_TASK_FIELDS
    )
    assert metrics["classification"]["summary"]["tasks_with_samples"] == 8
    assert metrics["classification"]["summary"]["accuracy"] == 1.0
    for task_metrics in metrics["classification"]["by_task"].values():
        assert task_metrics["accuracy"] == 1.0
    children = metrics["classification"]["by_task"]["taxonomy_level2_to_children"]
    assert children["set_exact_accuracy"] == 1.0
    assert children["micro_f1"] == 1.0


def test_classification_metrics_include_confusion_matrix() -> None:
    rows = [
        {
            "task": "settlement_classification",
            "gold_json": {"entity": "当天结", "category": "日结"},
            "prediction_json": {"entity": "当天结", "category": "月结"},
        }
    ]

    task_metrics = calculate_metrics(rows)["classification"]["by_task"][
        "settlement_classification"
    ]

    assert task_metrics["accuracy"] == 0.0
    assert task_metrics["per_field_accuracy"]["category"] == 0.0
    assert task_metrics["confusion_matrix"]["category=日结"]["category=月结"] == 1


def test_evaluate_model_generates_in_batches(monkeypatch, tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": f"sample:{index}",
            "prompt_messages": [{"role": "user", "content": str(index)}],
            "gold": '{"value": 1}',
            "gold_json": {"value": 1},
        }
        for index in range(5)
    ]
    observed_batch_sizes = []
    observed_adapter_paths = []

    monkeypatch.setattr(
        model_evaluation, "load_evaluation_rows", lambda dataset_path, limit=0: rows
    )

    def fake_load_model(model_path, adapter_path, torch_dtype):
        observed_adapter_paths.append(adapter_path)
        return object(), object(), "cpu"

    monkeypatch.setattr(model_evaluation, "load_model", fake_load_model)

    def fake_generate_batch(model, tokenizer, message_batches, max_new_tokens):
        observed_batch_sizes.append(len(message_batches))
        return ['{"value": 1}'] * len(message_batches)

    monkeypatch.setattr(model_evaluation, "generate_batch", fake_generate_batch)

    metrics = model_evaluation.evaluate_model(
        Path("dataset.jsonl"),
        tmp_path,
        model_path=Path("model"),
        batch_size=2,
    )

    assert observed_batch_sizes == [2, 2, 1]
    assert observed_adapter_paths == [None]
    assert metrics["object_exact_rate"] == 1.0
    assert metrics["prediction_performance"]["samples"] == 5
    assert metrics["prediction_performance"]["batch_size"] == 2
    assert metrics["prediction_performance"]["elapsed_seconds"] >= 0
    assert metrics["prediction_performance"]["samples_per_second"] > 0
    assert (tmp_path / "predictions.jsonl").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "classification_metrics.md").exists()


def test_evaluate_model_rejects_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        model_evaluation.evaluate_model(
            Path("dataset.jsonl"), tmp_path, batch_size=0, gold_as_prediction=True
        )
