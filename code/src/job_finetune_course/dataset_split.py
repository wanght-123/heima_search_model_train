"""按数据来源切分 train/dev/diagnostic/frozen_test。"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data_processing import write_jsonl
from .project_config import SPLIT_NAMES


def source_group(row: dict[str, Any]) -> str:
    """返回样本的来源组。

    同一 source_id 可能派生多个任务，必须分到同一个集合，避免数据泄漏。
    """

    return str(row.get("source_id") or row.get("sample_id"))


def split_by_source(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.80,
    dev_ratio: float = 0.05,
    diagnostic_ratio: float = 0.10,
    seed: int = 20260614,
) -> dict[str, list[dict[str, Any]]]:
    """以“来源组”为单位做可复现的随机切分。"""

    frozen_ratio = 1.0 - train_ratio - dev_ratio - diagnostic_ratio
    if min(train_ratio, dev_ratio, diagnostic_ratio, frozen_ratio) <= 0:
        raise ValueError("四个集合的比例都必须大于 0，且总和为 1")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[source_group(row)].append(row)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    total_groups = len(group_ids)
    train_end = int(total_groups * train_ratio)
    dev_end = train_end + int(total_groups * dev_ratio)
    diagnostic_end = dev_end + int(total_groups * diagnostic_ratio)

    group_splits = {
        "train": group_ids[:train_end],
        "dev": group_ids[train_end:dev_end],
        "diagnostic": group_ids[dev_end:diagnostic_end],
        "frozen_test": group_ids[diagnostic_end:],
    }
    return {
        split: [row for group_id in ids for row in groups[group_id]]
        for split, ids in group_splits.items()
    }


def save_splits(
    output_dir: Path, splits: dict[str, list[dict[str, Any]]], *, seed: int
) -> dict[str, Any]:
    """按 <task>/<split>.jsonl 保存切分结果和 manifest。"""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录必须为空：{output_dir}")

    task_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for split in SPLIT_NAMES:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in splits[split]:
            by_task[str(row["task"])].append(row)
        for task, task_rows in by_task.items():
            task_counts[task][split] = write_jsonl(
                output_dir / task / f"{split}.jsonl", task_rows
            )

    # 再次检查：任意两个 split 不得共用 source_id。
    groups = {
        split: {source_group(row) for row in rows} for split, rows in splits.items()
    }
    leakage = sum(
        len(groups[left] & groups[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )
    manifest = {
        "seed": seed,
        "split_rows": {name: len(splits[name]) for name in SPLIT_NAMES},
        "split_groups": {name: len(groups[name]) for name in SPLIT_NAMES},
        "task_counts": dict(task_counts),
        "task_total": dict(
            Counter(str(row["task"]) for rows in splits.values() for row in rows)
        ),
        "source_group_leakage": leakage,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
