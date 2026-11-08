#!/usr/bin/env python3
"""Prepare input-only batches for Codex-authored DPO generation.

This script selects and packages inputs, error targets, and authoritative task
instructions. It does not generate chosen/rejected content and never exposes an
SFT output field to Codex.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "code/data/datasets/dpo/codex_direct_v2_full1000/_generation"
SEEDS_DIR = GEN / "batches"
PROMPTS_DIR = GEN / "prompts"
OUTPUTS_DIR = GEN / "outputs"
LOGS_DIR = GEN / "logs"
SFT_ROOT = ROOT / "code/data/gemini_labeled_v2/tasks_fixed"
PILOT = ROOT / "code/data/datasets/dpo/codex_direct_v2_pilot10/pilot10_codex_direct_review.jsonl"
PROMPT_FILE = ROOT / "code/data/task_info/label_prompts_v2.md"
JOB_CATE_FILE = ROOT / "job_cate.json"
FROZEN = ROOT / "code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl"
RNG = random.Random(20260714)

REMAINING = {
    "entity_extraction": 398,
    "job_name_classification": 319,
    "work_time_classification": 69,
    "workplace_classification": 59,
    "benefit_classification": 59,
    "work_type_classification": 24,
    "settlement_classification": 24,
    "taxonomy_leaf_to_parent": 19,
    "taxonomy_level2_to_children": 19,
}

ENTITY_ERRORS = {
    "E01_boundary_too_short": 89,
    "E02_boundary_too_long": 90,
    "E04_field_confusion": 90,
    "E07_hallucination": 59,
    "E08_omission": 40,
    "E11_json_schema": 20,
    "E10_rule_exception": 10,
}
JOB_ERRORS = {
    "E05_category_confusion": 208,
    "E06_hierarchy_inconsistency": 48,
    "E09_context_ignored": 31,
    "E07_hallucination": 16,
    "E11_json_schema": 16,
}
SMALL_ERRORS = {
    "work_time_classification": {
        "E05_category_confusion": 50, "E10_rule_exception": 10, "E11_json_schema": 9,
    },
    "workplace_classification": {
        "E05_category_confusion": 45, "E10_rule_exception": 8, "E11_json_schema": 6,
    },
    "benefit_classification": {
        "E05_category_confusion": 35, "E03_over_extraction": 15,
        "E07_hallucination": 5, "E11_json_schema": 4,
    },
    # One E05 row is supplied by the successful smoke generation.
    "work_type_classification": {
        "E05_category_confusion": 17, "E10_rule_exception": 4, "E11_json_schema": 2,
    },
    # One E10 row is supplied by the successful smoke generation.
    "settlement_classification": {
        "E05_category_confusion": 12, "E10_rule_exception": 7, "E11_json_schema": 4,
    },
}
CHUNK_SIZE = {
    "entity_extraction": 20,
    "job_name_classification": 32,
    "work_time_classification": 35,
    "workplace_classification": 30,
    "benefit_classification": 30,
    "work_type_classification": 23,
    "settlement_classification": 23,
    "taxonomy_leaf_to_parent": 19,
    "taxonomy_level2_to_children": 19,
}
TARGET_FIELD = {
    "E01_boundary_too_short": "model_selected_field",
    "E02_boundary_too_long": "model_selected_field",
    "E03_over_extraction": "model_selected_field",
    "E04_field_confusion": "model_selected_fields",
    "E05_category_confusion": "category",
    "E06_hierarchy_inconsistency": "level1|level2|level3",
    "E07_hallucination": "model_selected_field",
    "E08_omission": "model_selected_field",
    "E09_context_ignored": "level1|level2|level3",
    "E10_rule_exception": "category_or_rule_field",
    "E11_json_schema": "json",
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def expand_errors(counts: dict[str, int]) -> list[str]:
    values = [error for error, count in counts.items() for _ in range(count)]
    RNG.shuffle(values)
    return values


def sft_inputs(task: str, used: set[str], frozen: set[str], frozen_norm: set[str], max_len: int) -> list[dict]:
    result = []
    for row in read_jsonl(SFT_ROOT / task / "train.jsonl"):
        # Deliberately select only input and source identity. row["output"] is never accessed.
        value = str(row["input"])
        if value in used or value in frozen or norm(value) in frozen_norm or not (4 <= len(value) <= max_len):
            continue
        result.append({"source_id": str(row.get("source_id", "")), "input": value})
    RNG.shuffle(result)
    return result


def seed(task: str, source: dict, error: str, index: int, source_type: str = "sft_input") -> dict:
    return {
        "seed_id": f"DIRECT_NEW_{index:04d}",
        "task": task,
        "source_type": source_type,
        "source_id": source["source_id"],
        "input": source["input"],
        "primary_error_type": error,
        "target_field": TARGET_FIELD[error],
    }


def extract_prompts() -> dict[str, str]:
    text = PROMPT_FILE.read_text(encoding="utf-8").split("## 标签生产提示词", 1)[1].split("## 微调训练提示词", 1)[0]
    prompts = dict(re.findall(r"### `([^`]+)`\s*\n\n```text\n(.*?)\n```", text, re.S))
    # job_cate.json is the sole category authority; avoid duplicating 1,058 paths in every call.
    job = prompts["job_name_classification"].split("候选分类体系（必须严格遵守）：", 1)[0].rstrip()
    prompts["job_name_classification"] = job + "\n全部合法职位路径唯一取自同目录 job_cate.json。"
    prompts["taxonomy_leaf_to_parent"] += "\n标准分类树唯一取自同目录 job_cate.json。"
    prompts["taxonomy_level2_to_children"] += "\n标准分类树唯一取自同目录 job_cate.json。"
    return prompts


def main() -> None:
    for directory in (SEEDS_DIR, PROMPTS_DIR, OUTPUTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for path in SEEDS_DIR.glob("*.json"):
        path.unlink()

    pilot_rows = list(read_jsonl(PILOT))
    used = {str(row["input"]) for row in pilot_rows}
    frozen, frozen_norm = set(), set()
    for row in read_jsonl(FROZEN):
        value = next((str(m.get("value", "")) for m in row.get("conversations", []) if m.get("from") == "human"), "")
        frozen.add(value)
        frozen_norm.add(norm(value))

    prompts = extract_prompts()
    for task, prompt in prompts.items():
        (PROMPTS_DIR / f"{task}.md").write_text(prompt.rstrip() + "\n", encoding="utf-8")
    shutil.copyfile(JOB_CATE_FILE, GEN / "job_cate.json")

    seeds: list[dict] = []
    serial = 1

    # Entity: reserve rich job posts for long-boundary and field-confusion negatives.
    rows = sft_inputs("entity_extraction", used, frozen, frozen_norm, 1200)
    jobs = [row for row in rows if row["input"].startswith("职位名称：")]
    queries = [row for row in rows if row["input"].startswith("用户搜索词：")]
    rich_errors = ["E02_boundary_too_long"] * 90 + ["E04_field_confusion"] * 90 + ["E01_boundary_too_short"] * 18
    query_errors = ["E01_boundary_too_short"] * 71 + ["E07_hallucination"] * 59 + ["E08_omission"] * 40 + ["E11_json_schema"] * 20 + ["E10_rule_exception"] * 10
    RNG.shuffle(rich_errors)
    RNG.shuffle(query_errors)
    for source, error in zip(jobs[:198], rich_errors):
        seeds.append(seed("entity_extraction", source, error, serial)); used.add(source["input"]); serial += 1
    for source, error in zip(queries[:200], query_errors):
        seeds.append(seed("entity_extraction", source, error, serial)); used.add(source["input"]); serial += 1

    # Job classification.
    rows = sft_inputs("job_name_classification", used, frozen, frozen_norm, 1000)
    for source, error in zip(rows[:319], expand_errors(JOB_ERRORS)):
        seeds.append(seed("job_name_classification", source, error, serial)); used.add(source["input"]); serial += 1

    # Remaining closed-set tasks. Smoke rows already supply one work-type and one settlement pair.
    for task in ("work_time_classification", "workplace_classification", "benefit_classification", "work_type_classification", "settlement_classification"):
        count = sum(SMALL_ERRORS[task].values())
        rows = sft_inputs(task, used, frozen, frozen_norm, 140)
        for source, error in zip(rows[:count], expand_errors(SMALL_ERRORS[task])):
            seeds.append(seed(task, source, error, serial)); used.add(source["input"]); serial += 1

    # Taxonomy-only inputs, excluding the approved Pilot cases.
    job_cate = json.loads(JOB_CATE_FILE.read_text(encoding="utf-8"))
    leaf_paths: dict[str, list[tuple[str, str, str]]] = {}
    for level1, children in job_cate.items():
        for level2, leaves in children.items():
            for leaf in leaves:
                leaf_paths.setdefault(leaf, []).append((level1, level2, leaf))
    unique_leaves = [leaf for leaf, paths in leaf_paths.items() if len(paths) == 1 and leaf != "量化研究员"]
    RNG.shuffle(unique_leaves)
    for leaf in unique_leaves[:19]:
        source = {"source_id": "job_cate.json", "input": f"三级职位：{leaf}"}
        seeds.append(seed("taxonomy_leaf_to_parent", source, "E06_hierarchy_inconsistency", serial, "taxonomy")); serial += 1
    paths = [(level1, level2) for level1, children in job_cate.items() for level2 in children if (level1, level2) != ("医疗健康", "临床试验")]
    RNG.shuffle(paths)
    child_errors = expand_errors({"E08_omission": 6, "E03_over_extraction": 7, "E06_hierarchy_inconsistency": 3, "E11_json_schema": 3})
    for (level1, level2), error in zip(paths[:19], child_errors):
        source = {"source_id": "job_cate.json", "input": f"一级职位：{level1}\n二级职位：{level2}"}
        seeds.append(seed("taxonomy_level2_to_children", source, error, serial, "taxonomy")); serial += 1

    if len(seeds) != 988:
        raise ValueError(f"expected 988 prepared seeds, got {len(seeds)}")

    batches = []
    for task in REMAINING:
        task_rows = [row for row in seeds if row["task"] == task]
        size = CHUNK_SIZE[task]
        for offset in range(0, len(task_rows), size):
            batch_rows = task_rows[offset:offset + size]
            batch_id = f"{task}_{offset // size + 1:03d}"
            path = SEEDS_DIR / f"{batch_id}.json"
            path.write_text(json.dumps(batch_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            batches.append({"batch_id": batch_id, "task": task, "count": len(batch_rows), "seed_file": path.name})

    manifest = {
        "dataset_target": 1000,
        "approved_pilot": 10,
        "successful_smoke_pairs": 2,
        "prepared_batch_pairs": len(seeds),
        "sft_output_used": False,
        "generation_model": "Codex CLI authenticated with ChatGPT; actual model recorded per run log",
        "task_remaining_targets": REMAINING,
        "prepared_task_counts": dict(Counter(row["task"] for row in seeds)),
        "prepared_error_counts": dict(Counter(row["primary_error_type"] for row in seeds)),
        "batch_count": len(batches),
        "batches": batches,
        "authorities": {
            "label_prompts_sha256": hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest(),
            "job_cate_sha256": hashlib.sha256(JOB_CATE_FILE.read_bytes()).hexdigest(),
        },
    }
    (GEN / "seed_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": len(seeds), "batches": len(batches), "tasks": manifest["prepared_task_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
