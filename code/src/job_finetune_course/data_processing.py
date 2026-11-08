"""数据读取、基础检查和 LLaMA-Factory 格式转换。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .project_config import SPLIT_NAMES, TASK_NAMES


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL：每一行是一个 JSON 对象。"""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 格式错误：{path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """写入 JSONL，返回写入的行数。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_task_rows(
    input_dir: Path, *, split: str = "train", limit_per_task: int = 0
) -> list[dict[str, Any]]:
    """合并九个任务的数据。

    目录结构约定为：<input_dir>/<task>/<split>.jsonl。
    """

    rows: list[dict[str, Any]] = []
    for task in TASK_NAMES:
        path = input_dir / task / f"{split}.jsonl"
        if not path.exists():
            continue
        task_rows = read_jsonl(path)
        if limit_per_task > 0:
            task_rows = task_rows[:limit_per_task]
        for row in task_rows:
            # 某些历史数据没有 task 字段，可以由所在目录确定。
            row.setdefault("task", task)
        rows.extend(task_rows)
    if not rows:
        raise ValueError(f"没有找到数据：{input_dir} split={split}")
    return rows


def validate_row(row: dict[str, Any]) -> list[str]:
    """检查一条样本的最小必需字段。"""

    errors: list[str] = []
    if not row.get("sample_id"):
        errors.append("缺少 sample_id")
    if row.get("task") not in TASK_NAMES:
        errors.append("task 不在九任务列表中")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages 至少需要 user 和 assistant 消息")
        return errors
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if not roles or roles[-1] != "assistant":
        errors.append("最后一条 message 必须是 assistant 答案")
    try:
        json.loads(str(messages[-1].get("content", "")))
    except (AttributeError, json.JSONDecodeError):
        errors.append("assistant 答案不是合法 JSON")
    return errors


def build_data_report(
    rows: list[dict[str, Any]], *, max_error_examples: int = 20
) -> dict[str, Any]:
    """生成易读的数据检查报告。"""

    error_examples = []
    for row in rows:
        errors = validate_row(row)
        if errors and len(error_examples) < max_error_examples:
            error_examples.append({"sample_id": row.get("sample_id"), "errors": errors})
    return {
        "total": len(rows),
        "task_counts": dict(Counter(str(row.get("task")) for row in rows)),
        "error_examples": error_examples,
        "passed": not error_examples,
    }


def to_sharegpt(row: dict[str, Any]) -> dict[str, Any]:
    """将项目的 messages 转成 LLaMA-Factory ShareGPT 格式。"""

    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    conversations = []
    for message in row["messages"]:
        role = role_map.get(message["role"])
        if role is None:
            raise ValueError(f"不支持的消息角色：{message['role']}")
        conversations.append({"from": role, "value": str(message.get("content", ""))})
    return {"conversations": conversations}


def export_llamafactory_dataset(
    input_dir: Path, output_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """将四个 split 分别合并成 LLaMA-Factory 训练文件。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info: dict[str, Any] = {}
    report: dict[str, Any] = {"splits": {}}
    for split in SPLIT_NAMES:
        rows = load_task_rows(input_dir, split=split)
        name = f"recruitment_sft_{split}"
        file_name = f"{name}.jsonl"
        output_path = output_dir / file_name
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"输出已存在，如需覆盖请增加 --overwrite：{output_path}"
            )
        count = write_jsonl(output_path, (to_sharegpt(row) for row in rows))
        dataset_info[name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        }
        report["splits"][split] = count
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
