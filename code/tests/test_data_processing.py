"""数据处理模块的最小测试。"""

import json

from job_finetune_course.data_processing import (
    build_data_report,
    export_llamafactory_dataset,
    read_jsonl,
    to_sharegpt,
    write_jsonl,
)
from job_finetune_course.project_config import SPLIT_NAMES


def sample_row() -> dict:
    return {
        "sample_id": "sample-1",
        "source_id": "source-1",
        "task": "entity_extraction",
        "messages": [
            {"role": "system", "content": "按 JSON 输出"},
            {"role": "user", "content": "招聘 Java 工程师"},
            {"role": "assistant", "content": json.dumps({"position": ["Java 工程师"]})},
        ],
    }


def test_report_and_sharegpt_conversion() -> None:
    row = sample_row()
    report = build_data_report([row])
    converted = to_sharegpt(row)
    assert report["passed"] is True
    assert report["task_counts"] == {"entity_extraction": 1}
    assert [message["from"] for message in converted["conversations"]] == [
        "system",
        "human",
        "gpt",
    ]


def test_export_four_splits(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    for split in SPLIT_NAMES:
        write_jsonl(input_dir / "entity_extraction" / f"{split}.jsonl", [sample_row()])
    report = export_llamafactory_dataset(input_dir, output_dir)
    assert report["splits"] == {split: 1 for split in SPLIT_NAMES}
    assert len(read_jsonl(output_dir / "recruitment_sft_train.jsonl")) == 1
