#!/usr/bin/env python3
"""Prepare source-only shards for Codex-based relabeling.

This script deliberately does not generate labels. It strips old assistant
answers from worker inputs so each shard must be labeled from system + human.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_HINTS = [
    ("benefit_classification", ["福利标准化专家", "standardBenefit"]),
    ("job_name_classification", ["岗位名称分类专家", "level3"]),
    ("work_time_classification", ["工作时间分类专家", "fullCategory"]),
    ("workplace_classification", ["工作场所分类专家", "rawWorkPlace"]),
    ("work_type_classification", ["用工类型分类专家", "employmentType"]),
    ("settlement_classification", ["工作酬劳结算方式分类专家", "category"]),
    ("entity_extraction", ["招聘文本实体抽取标注专家", "position"]),
    ("taxonomy_leaf_to_parent", ["职位分类树关系专家", "level3"]),
]


def infer_task(system_text: str) -> str:
    for task, hints in TASK_HINTS:
        if all(hint in system_text for hint in hints):
            return task
    for task, hints in TASK_HINTS:
        if any(hint in system_text for hint in hints):
            return task
    return "unknown"


def flatten_job_paths(job_cate_path: Path) -> list[str]:
    data = json.loads(job_cate_path.read_text(encoding="utf-8"))
    paths: list[str] = []
    for level1, level2_map in data.items():
        for level2, leaves in level2_map.items():
            for leaf in leaves:
                paths.append(f"{level1} / {level2} / {leaf}")
    return sorted(paths)


def prepare_shards(dataset_path: Path, job_cate_path: Path, out_dir: Path, shard_size: int) -> None:
    source_dir = out_dir / "source_shards"
    source_dir.mkdir(parents=True, exist_ok=True)

    paths = flatten_job_paths(job_cate_path)
    (out_dir / "job_paths.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")

    manifest = {
        "dataset_path": str(dataset_path),
        "job_cate_path": str(job_cate_path),
        "shard_size": shard_size,
        "shards": [],
    }

    shard_file = None
    shard_index = -1
    shard_count = 0
    total = 0
    task_counts: dict[str, int] = {}

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for source_line, raw in enumerate(handle, start=1):
                if (source_line - 1) % shard_size == 0:
                    if shard_file:
                        shard_file.close()
                        manifest["shards"][-1]["count"] = shard_count
                    shard_index += 1
                    shard_count = 0
                    shard_path = source_dir / f"shard-{shard_index:04d}.jsonl"
                    shard_file = shard_path.open("w", encoding="utf-8")
                    manifest["shards"].append(
                        {
                            "shard": f"shard-{shard_index:04d}",
                            "path": str(shard_path),
                            "start_line": source_line,
                            "end_line": None,
                            "count": None,
                        }
                    )

                row = json.loads(raw)
                conv = row["conversations"]
                system = next(item["value"] for item in conv if item["from"] == "system")
                human = next(item["value"] for item in conv if item["from"] == "human")
                task = infer_task(system)
                task_counts[task] = task_counts.get(task, 0) + 1

                shard_file.write(
                    json.dumps(
                        {
                            "source_line": source_line,
                            "task": task,
                            "system": system,
                            "human": human,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                shard_count += 1
                total += 1
                manifest["shards"][-1]["end_line"] = source_line
    finally:
        if shard_file:
            shard_file.close()
            manifest["shards"][-1]["count"] = shard_count

    manifest["total_rows"] = total
    manifest["total_shards"] = len(manifest["shards"])
    manifest["task_counts"] = dict(sorted(task_counts.items()))
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--job-cate", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=250)
    args = parser.parse_args()

    prepare_shards(args.dataset, args.job_cate, args.out_dir, args.shard_size)


if __name__ == "__main__":
    main()
