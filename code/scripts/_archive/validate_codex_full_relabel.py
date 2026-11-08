#!/usr/bin/env python3
"""Validate and optionally merge Codex-produced relabel shards.

This script checks structure, candidate membership, ordering, and source
alignment. It does not create or modify labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED_ENTITY_KEYS = {
    "position": list,
    "company": str,
    "location": str,
    "workPlace": str,
    "workMode": str,
    "salary": str,
    "salaryPayment": str,
    "employmentType": str,
    "schedule": list,
    "description": list,
    "responsibilities": list,
    "requirements": list,
    "preferredQualifications": list,
    "education": str,
    "workExperience": str,
    "benefits": list,
}


def infer_task(system_text: str) -> str:
    if "招聘文本实体抽取标注专家" in system_text:
        return "entity_extraction"
    if "岗位名称分类专家" in system_text:
        return "job_name_classification"
    if "员工福利标准化专家" in system_text:
        return "benefit_classification"
    if "工作时间分类专家" in system_text:
        return "work_time_classification"
    if "工作场所分类专家" in system_text:
        return "workplace_classification"
    if "职位分类树关系专家" in system_text:
        return "taxonomy_leaf_to_parent"
    if "用工类型分类专家" in system_text:
        return "work_type_classification"
    if "工作酬劳结算方式分类专家" in system_text:
        return "settlement_classification"
    return "unknown"


def load_job_paths(job_cate_path: Path) -> tuple[set[tuple[str, str, str]], set[str], set[tuple[str, str]]]:
    data = json.loads(job_cate_path.read_text(encoding="utf-8"))
    triples: set[tuple[str, str, str]] = set()
    level1s: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for level1, level2_map in data.items():
        level1s.add(level1)
        for level2, leaves in level2_map.items():
            pairs.add((level1, level2))
            for leaf in leaves:
                triples.add((level1, level2, leaf))
    return triples, level1s, pairs


def candidates_from_system(system_text: str) -> set[str]:
    marker = "候选分类体系"
    if marker not in system_text:
        return set()
    tail = system_text.split(marker, 1)[1]
    match = re.search(r"\[[\s\S]*?\]", tail)
    if not match:
        return set()
    return set(json.loads(match.group(0)))


def assert_keys(obj: dict, keys: set[str], errors: list[str], where: str) -> None:
    if set(obj) != keys:
        errors.append(f"{where}: keys {sorted(obj)} != {sorted(keys)}")


def validate_label(
    task: str,
    label: dict,
    system: str,
    human: str,
    job_triples: set[tuple[str, str, str]],
    job_pairs: set[tuple[str, str]],
    errors: list[str],
    where: str,
) -> None:
    candidates = candidates_from_system(system)
    if task == "benefit_classification":
        assert_keys(label, {"entity", "standardBenefit"}, errors, where)
        if label.get("standardBenefit") not in candidates:
            errors.append(f"{where}: invalid standardBenefit {label.get('standardBenefit')!r}")
    elif task == "job_name_classification":
        assert_keys(label, {"position", "level1", "level2", "level3"}, errors, where)
        triple = (label.get("level1"), label.get("level2"), label.get("level3"))
        if triple not in job_triples:
            errors.append(f"{where}: invalid job path {triple!r}")
    elif task == "work_time_classification":
        assert_keys(label, {"entity", "level1", "level2", "fullCategory"}, errors, where)
        full = label.get("fullCategory")
        if full == "其他":
            if label.get("level1") or label.get("level2"):
                errors.append(f"{where}: other work_time should have empty parents")
        elif full not in candidates or full != f"{label.get('level1')}-{label.get('level2')}":
            errors.append(f"{where}: invalid work_time category {full!r}")
    elif task == "workplace_classification":
        assert_keys(label, {"rawWorkPlace", "level1", "level2"}, errors, where)
        level1 = label.get("level1")
        level2 = label.get("level2")
        if not level1 or not level2:
            errors.append(f"{where}: workplace level1/level2 cannot be empty")
        else:
            level1_candidates = {item.split("-", 1)[0] for item in candidates if "-" in item}
            if level1 not in level1_candidates:
                errors.append(f"{where}: invalid workplace level1 {level1!r}")
    elif task == "work_type_classification":
        assert_keys(label, {"entity", "category"}, errors, where)
        if label.get("category") not in candidates:
            errors.append(f"{where}: invalid work_type category {label.get('category')!r}")
    elif task == "settlement_classification":
        assert_keys(label, {"entity", "category"}, errors, where)
        if label.get("category") not in candidates:
            errors.append(f"{where}: invalid settlement category {label.get('category')!r}")
    elif task == "entity_extraction":
        assert_keys(label, set(EXPECTED_ENTITY_KEYS), errors, where)
        for key, typ in EXPECTED_ENTITY_KEYS.items():
            if key in label and not isinstance(label[key], typ):
                errors.append(f"{where}: entity field {key} must be {typ.__name__}")
        if label.get("workMode") not in {"", "现场办公", "远程办公", "线上线下结合"}:
            errors.append(f"{where}: invalid workMode {label.get('workMode')!r}")
    elif task == "taxonomy_leaf_to_parent":
        assert_keys(label, {"level3", "level2", "level1"}, errors, where)
        triple = (label.get("level1"), label.get("level2"), label.get("level3"))
        if triple not in job_triples:
            errors.append(f"{where}: invalid taxonomy path {triple!r}")
        if (label.get("level1"), label.get("level2")) not in job_pairs:
            errors.append(f"{where}: invalid taxonomy parent {triple[:2]!r}")
    else:
        errors.append(f"{where}: unknown task")


def validate(base_dir: Path, job_cate_path: Path, merge_path: Path | None, require_full: bool) -> int:
    manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    source_dir = base_dir / "source_shards"
    relabel_dir = base_dir / "relabel_shards"
    audit_dir = base_dir / "audit_shards"
    job_triples, _job_level1s, job_pairs = load_job_paths(job_cate_path)

    errors: list[str] = []
    validated_rows = 0
    completed_shards: list[str] = []
    merged_rows: list[str] = []

    for shard in manifest["shards"]:
        name = shard["shard"]
        source_path = source_dir / f"{name}.jsonl"
        relabel_path = relabel_dir / f"{name}.jsonl"
        audit_path = audit_dir / f"{name}.review.jsonl"

        if not relabel_path.exists() or not audit_path.exists():
            if require_full:
                errors.append(f"{name}: missing output")
            continue

        source_rows = [json.loads(line) for line in source_path.open(encoding="utf-8")]
        relabel_lines = relabel_path.read_text(encoding="utf-8").splitlines()
        audit_rows = [json.loads(line) for line in audit_path.open(encoding="utf-8")]

        if len(relabel_lines) != shard["count"]:
            errors.append(f"{name}: relabel line count {len(relabel_lines)} != {shard['count']}")
        if len(audit_rows) != shard["count"]:
            errors.append(f"{name}: audit line count {len(audit_rows)} != {shard['count']}")

        for idx, (src, raw_out) in enumerate(zip(source_rows, relabel_lines), start=1):
            where = f"{name}:{idx}/source_line={src['source_line']}"
            try:
                out = json.loads(raw_out)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSONL {exc}")
                continue
            conv = out.get("conversations")
            if not isinstance(conv, list) or len(conv) != 3:
                errors.append(f"{where}: conversations must have 3 turns")
                continue
            if conv[0] != {"from": "system", "value": src["system"]}:
                errors.append(f"{where}: system turn mismatch")
            if conv[1] != {"from": "human", "value": src["human"]}:
                errors.append(f"{where}: human turn mismatch")
            if conv[2].get("from") != "gpt" or not isinstance(conv[2].get("value"), str):
                errors.append(f"{where}: gpt turn invalid")
                continue
            try:
                label = json.loads(conv[2]["value"])
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: gpt.value is not JSON {exc}")
                continue
            task = infer_task(src["system"])
            validate_label(
                task,
                label,
                src["system"],
                src["human"],
                job_triples,
                job_pairs,
                errors,
                where,
            )
            if idx <= len(audit_rows):
                audit = audit_rows[idx - 1]
                if audit.get("source_line") != src["source_line"]:
                    errors.append(f"{where}: audit source_line mismatch")
                if audit.get("gpt") != conv[2]["value"]:
                    errors.append(f"{where}: audit gpt mismatch")
            merged_rows.append(raw_out)
            validated_rows += 1
        completed_shards.append(name)

    if merge_path and not errors:
        merge_path.write_text("\n".join(merged_rows) + ("\n" if merged_rows else ""), encoding="utf-8")

    print(json.dumps(
        {
            "completed_shards": len(completed_shards),
            "validated_rows": validated_rows,
            "errors": len(errors),
            "first_errors": errors[:20],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--job-cate", required=True, type=Path)
    parser.add_argument("--merge-path", type=Path)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    raise SystemExit(validate(args.base_dir, args.job_cate, args.merge_path, args.require_full))


if __name__ == "__main__":
    main()
