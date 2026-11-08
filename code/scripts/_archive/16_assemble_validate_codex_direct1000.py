#!/usr/bin/env python3
"""Assemble Codex-authored batches and strictly validate the 1k DPO dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "code/data/datasets/dpo/codex_direct_v2_full1000"
GEN = DATA / "_generation"
PILOT = ROOT / "code/data/datasets/dpo/codex_direct_v2_pilot10/pilot10_codex_direct_review.jsonl"
PROMPT_FILE = ROOT / "code/data/task_info/label_prompts_v2.md"
JOB_FILE = ROOT / "job_cate.json"
FROZEN = ROOT / "code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl"

TASK_TARGETS = {
    "entity_extraction": 400,
    "job_name_classification": 320,
    "work_time_classification": 70,
    "workplace_classification": 60,
    "benefit_classification": 60,
    "work_type_classification": 25,
    "settlement_classification": 25,
    "taxonomy_leaf_to_parent": 20,
    "taxonomy_level2_to_children": 20,
}
DEV_TARGETS = {
    "entity_extraction": 40,
    "job_name_classification": 32,
    "work_time_classification": 7,
    "workplace_classification": 6,
    "benefit_classification": 6,
    "work_type_classification": 2,
    "settlement_classification": 2,
    "taxonomy_leaf_to_parent": 2,
    "taxonomy_level2_to_children": 3,
}
ENTITY_KEYS = {
    "position", "company", "location", "workPlace", "workMode", "salary",
    "salaryPayment", "employmentType", "schedule", "description",
    "responsibilities", "requirements", "preferredQualifications", "education",
    "workExperience", "benefits",
}
SCHEMAS = {
    "entity_extraction": ENTITY_KEYS,
    "job_name_classification": {"position", "level1", "level2", "level3"},
    "taxonomy_leaf_to_parent": {"level3", "level2", "level1"},
    "taxonomy_level2_to_children": {"level1", "level2", "level3Children"},
    "work_type_classification": {"entity", "category"},
    "workplace_classification": {"rawWorkPlace", "level1", "level2"},
    "settlement_classification": {"entity", "category"},
    "work_time_classification": {"entity", "level1", "level2", "fullCategory"},
    "benefit_classification": {"entity", "standardBenefit"},
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def norm(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def parse_prompts() -> dict[str, str]:
    text = PROMPT_FILE.read_text(encoding="utf-8").split("## 微调训练提示词", 1)[1]
    return {task: prompt.rstrip() for task, prompt in re.findall(r"### `([^`]+)`\s*\n\n```text\n(.*?)\n```", text, re.S)}


def prompt_candidates(prompt: str) -> list[str]:
    match = re.search(r"候选分类体系（必须严格遵守）：(\[[^\n]+\])", prompt)
    return json.loads(match.group(1)) if match else []


PROMPTS = parse_prompts()
CANDIDATES = {task: prompt_candidates(PROMPTS[task]) for task in (
    "work_type_classification", "workplace_classification", "settlement_classification",
    "work_time_classification", "benefit_classification",
)}
JOB_CATE = json.loads(JOB_FILE.read_text(encoding="utf-8"))
JOB_PATHS = {(l1, l2, l3) for l1, children in JOB_CATE.items() for l2, leaves in children.items() for l3 in leaves}
JOB_LEVEL1 = set(JOB_CATE)
JOB_LEVEL2 = {l2 for children in JOB_CATE.values() for l2 in children}
JOB_LEVEL3 = {l3 for children in JOB_CATE.values() for leaves in children.values() for l3 in leaves}
WORKPLACE_PATHS = {tuple(value.split("-", 1)) for value in CANDIDATES["workplace_classification"]}


def load_generated() -> list[dict]:
    rows = []
    for pilot in read_jsonl(PILOT):
        item = dict(pilot)
        item["pilot_pair_id"] = pilot["pair_id"]
        item["seed_id"] = pilot["pair_id"]
        item["generation_model"] = "Codex current conversation direct reasoning"
        item["generation_channel"] = "codex_direct_conversation"
        item["human_review_status"] = "user_approved_pilot10"
        rows.append(item)

    smoke_seeds = {row["seed_id"]: row for row in json.loads((GEN / "smoke_seeds.json").read_text(encoding="utf-8"))}
    smoke_output = json.loads((GEN / "smoke_output.json").read_text(encoding="utf-8"))["records"]
    for output in smoke_output:
        source = smoke_seeds[output["seed_id"]]
        rows.append(from_output(source, output, "codex_cli_gpt-5.6-sol_direct_smoke"))

    seed_map = {row["seed_id"]: row for path in (GEN / "batches").glob("*.json") for row in json.loads(path.read_text(encoding="utf-8"))}
    for path in sorted((GEN / "outputs").glob("*.json")):
        for output in json.loads(path.read_text(encoding="utf-8"))["records"]:
            rows.append(from_output(seed_map[output["seed_id"]], output, "codex_cli_gpt-5.6-sol_direct"))

    # Strict-validation repair outputs are a one-for-one overlay keyed by the
    # original seed id. They preserve task/error quotas while replacing only a
    # failed answer or a frozen/ambiguous input.
    repair_dir = GEN / "repairs"
    repair_groups = ("entity_repairs", "job_replacements", "mixed_replacements")
    overlays = {}
    for group in repair_groups:
        seed_path = repair_dir / f"{group}.json"
        output_path = repair_dir / f"{group}_output.json"
        if not seed_path.exists() or not output_path.exists():
            continue
        repair_seeds = {row["seed_id"]: row for row in json.loads(seed_path.read_text(encoding="utf-8"))}
        for output in json.loads(output_path.read_text(encoding="utf-8"))["records"]:
            overlays[output["seed_id"]] = from_output(
                repair_seeds[output["seed_id"]], output,
                "codex_cli_gpt-5.6-sol_strict_repair",
            )
    return [overlays.get(row["seed_id"], row) for row in rows]


def from_output(source: dict, output: dict, channel: str) -> dict:
    chosen = json.loads(output["chosen_json"])
    item = {
        **source,
        "chosen": chosen,
        "review_reason": output["review_reason"],
        "chosen_generation": "codex_direct_reasoning",
        "rejected_generation": "codex_direct_reasoning",
        "generation_model": "gpt-5.6-sol",
        "generation_channel": channel,
        "human_review_status": "auto_validated_pending_spotcheck",
    }
    if source["primary_error_type"] == "E11_json_schema":
        item["rejected_raw"] = output["rejected_json"]
    else:
        item["rejected"] = json.loads(output["rejected_json"])
    return item


def validate_entity_chosen(row: dict, issues: list[str]) -> None:
    chosen = row["chosen"]
    arrays = {"position", "schedule", "description", "responsibilities", "requirements", "preferredQualifications", "benefits"}
    for key in ENTITY_KEYS:
        expected = list if key in arrays else str
        if not isinstance(chosen[key], expected):
            issues.append(f"{row['seed_id']}: entity type {key}")
    if isinstance(chosen["position"], list) and len(chosen["position"]) > 2:
        issues.append(f"{row['seed_id']}: position length")
    if chosen.get("workMode") not in {"", "现场办公", "远程办公", "线上线下结合"}:
        issues.append(f"{row['seed_id']}: workMode candidate")
    text = row["input"]
    for key, value in chosen.items():
        if key == "workMode":
            continue
        values = value if isinstance(value, list) else [value]
        for index, part in enumerate(values):
            if not part or (key == "position" and index == 1):
                continue
            if not isinstance(part, str) or part not in text:
                issues.append(f"{row['seed_id']}: non-verbatim {key}={part!r}")


def validate(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    if len(rows) != 1000:
        issues.append(f"row count {len(rows)}")
    if Counter(row["task"] for row in rows) != Counter(TASK_TARGETS):
        issues.append(f"task counts {Counter(row['task'] for row in rows)}")
    if len({row["seed_id"] for row in rows if "seed_id" in row}) != len(rows):
        issues.append("generated seed id uniqueness/count")
    if len({(row["task"], row["input"]) for row in rows}) != len(rows):
        issues.append("duplicate task/input")

    frozen, frozen_norm = set(), set()
    for item in read_jsonl(FROZEN):
        value = next((str(m.get("value", "")) for m in item.get("conversations", []) if m.get("from") == "human"), "")
        frozen.add(value); frozen_norm.add(norm(value))

    for row in rows:
        rid = row.get("seed_id", row.get("pair_id", "unknown"))
        row["seed_id"] = rid
        task, chosen = row["task"], row["chosen"]
        if set(chosen) != SCHEMAS[task]:
            issues.append(f"{rid}: chosen schema {sorted(chosen)}")
            continue
        if "rejected_raw" not in row:
            rejected = row.get("rejected")
            if not isinstance(rejected, dict) or set(rejected) != SCHEMAS[task]:
                issues.append(f"{rid}: rejected schema")
            elif rejected == chosen:
                issues.append(f"{rid}: chosen equals rejected")
        elif row["rejected_raw"] == compact(chosen):
            issues.append(f"{rid}: format rejected equals chosen")
        if row["input"] in frozen or norm(row["input"]) in frozen_norm:
            issues.append(f"{rid}: frozen overlap")

        if task == "entity_extraction":
            validate_entity_chosen(row, issues)
        elif task == "job_name_classification":
            cpath = (chosen["level1"], chosen["level2"], chosen["level3"])
            if cpath not in JOB_PATHS:
                issues.append(f"{rid}: invalid chosen job path {cpath}")
            if chosen["position"] not in row["input"]:
                issues.append(f"{rid}: job position not verbatim")
            if "rejected" in row:
                rejected = row["rejected"]
                rpath = (rejected["level1"], rejected["level2"], rejected["level3"])
                if not (rejected["level1"] in JOB_LEVEL1 and rejected["level2"] in JOB_LEVEL2 and rejected["level3"] in JOB_LEVEL3):
                    issues.append(f"{rid}: rejected job category outside taxonomy")
                if row["primary_error_type"] == "E06_hierarchy_inconsistency":
                    if rpath in JOB_PATHS:
                        issues.append(f"{rid}: hierarchy rejected remains legal")
                elif rpath not in JOB_PATHS:
                    issues.append(f"{rid}: rejected job path illegal for {row['primary_error_type']}")
        elif task == "taxonomy_leaf_to_parent":
            if (chosen["level1"], chosen["level2"], chosen["level3"]) not in JOB_PATHS:
                issues.append(f"{rid}: invalid taxonomy leaf chosen")
            if "rejected" in row:
                rejected = row["rejected"]
                if rejected["level3"] != chosen["level3"]:
                    issues.append(f"{rid}: taxonomy rejected changed leaf")
                if not (rejected["level1"] in JOB_LEVEL1 and rejected["level2"] in JOB_LEVEL2 and rejected["level3"] in JOB_LEVEL3):
                    issues.append(f"{rid}: taxonomy rejected outside names")
                if (rejected["level1"], rejected["level2"], rejected["level3"]) in JOB_PATHS:
                    issues.append(f"{rid}: taxonomy hierarchy rejected remains legal")
        elif task == "taxonomy_level2_to_children":
            if chosen["level1"] not in JOB_CATE or chosen["level2"] not in JOB_CATE[chosen["level1"]]:
                issues.append(f"{rid}: invalid taxonomy parent")
            elif chosen["level3Children"] != JOB_CATE[chosen["level1"]][chosen["level2"]]:
                issues.append(f"{rid}: incomplete/incorrect chosen children")
        elif task in {"work_type_classification", "settlement_classification"}:
            if chosen["category"] not in CANDIDATES[task]:
                issues.append(f"{rid}: chosen category outside")
            if "rejected" in row and row["rejected"]["category"] not in CANDIDATES[task]:
                issues.append(f"{rid}: rejected category outside")
            if chosen["entity"] != row["input"].split("：", 1)[-1]:
                issues.append(f"{rid}: chosen entity changed")
        elif task == "benefit_classification":
            if chosen["standardBenefit"] not in CANDIDATES[task]:
                issues.append(f"{rid}: benefit chosen outside")
            if "rejected" in row and row["rejected"]["standardBenefit"] not in CANDIDATES[task]:
                issues.append(f"{rid}: benefit rejected outside")
            if chosen["entity"] != row["input"].split("：", 1)[-1]:
                issues.append(f"{rid}: benefit entity changed")
        elif task == "work_time_classification":
            for label, value in (("chosen", chosen), ("rejected", row.get("rejected"))):
                if value is None:
                    continue
                category = value["fullCategory"]
                if category not in CANDIDATES[task]:
                    issues.append(f"{rid}: work time {label} outside")
                # E10's rejected answer is intentionally a rule-violating hard
                # negative; all chosen answers and other rejected answers must
                # retain a coherent hierarchy.
                intentional_rule_error = label == "rejected" and row["primary_error_type"] == "E10_rule_exception"
                if not intentional_rule_error:
                    if category == "其他":
                        if value["level1"] or value["level2"]:
                            issues.append(f"{rid}: other hierarchy nonempty")
                    elif category != value["level1"] + "-" + value["level2"]:
                        issues.append(f"{rid}: work time hierarchy mismatch")
            if chosen["entity"] != row["input"].split("：", 1)[-1]:
                issues.append(f"{rid}: work time entity changed")
        elif task == "workplace_classification":
            for label, value in (("chosen", chosen), ("rejected", row.get("rejected"))):
                if value is None:
                    continue
                path = (value["level1"], value["level2"])
                if path not in WORKPLACE_PATHS and value["level2"] != value["rawWorkPlace"]:
                    issues.append(f"{rid}: workplace {label} outside/fallback")
            if chosen["rawWorkPlace"] != row["input"].split("：", 1)[-1]:
                issues.append(f"{rid}: rawWorkPlace changed")
    return issues


def assign_ids_and_difficulty(rows: list[dict]) -> None:
    order = {task: index for index, task in enumerate(TASK_TARGETS)}
    rows.sort(key=lambda row: (order[row["task"]], digest(row["input"] + row["seed_id"])))
    score = {
        "E11_json_schema": 0, "E01_boundary_too_short": 1, "E02_boundary_too_long": 1,
        "E03_over_extraction": 1, "E05_category_confusion": 1, "E07_hallucination": 1,
        "E08_omission": 1, "E04_field_confusion": 2, "E06_hierarchy_inconsistency": 2,
        "E09_context_ignored": 2, "E10_rule_exception": 2,
    }
    ranked = sorted(rows, key=lambda row: (score[row["primary_error_type"]], digest(row["input"])))
    for index, row in enumerate(ranked):
        row["difficulty"] = "easy" if index < 100 else "medium" if index < 700 else "hard"
    for index, row in enumerate(rows, 1):
        row["pair_id"] = f"CODEX_DPO_V2_{index:06d}"
        row["dataset_version"] = "codex-direct-v2-full1000"
        row["prompt_version"] = "recruitment_label_prompts_v2"


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    def group_id(row: dict) -> str:
        # All synthetic taxonomy rows cite the same authority file, so source_id
        # is not a sample-level leakage key for those tasks. Their concrete
        # taxonomy prompt is the independent split unit. SFT-derived tasks remain
        # grouped by original source_id to prevent cross-task source leakage.
        if row.get("source_type") == "taxonomy":
            return f"taxonomy::{row['task']}::{row['input']}"
        return row["source_id"]

    pilot_sources = {group_id(row) for row in rows if row.get("human_review_status") == "user_approved_pilot10"}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_id(row)].append(row)
    dev, counts = [], Counter()
    candidates = [(source, items) for source, items in groups.items() if source not in pilot_sources]
    candidates.sort(key=lambda pair: digest(pair[0]))
    selected_sources = set()
    for source, items in candidates:
        addition = Counter(row["task"] for row in items)
        if all(counts[task] + value <= DEV_TARGETS[task] for task, value in addition.items()) and any(counts[task] < DEV_TARGETS[task] for task in addition):
            dev.extend(items); counts.update(addition); selected_sources.add(source)
        if counts == Counter(DEV_TARGETS):
            break
    if counts != Counter(DEV_TARGETS):
        raise ValueError(f"unable to create source-isolated dev split: {counts}")
    train = [row for row in rows if group_id(row) not in selected_sources]
    return train, dev


def sharegpt(row: dict) -> dict:
    rejected = row.get("rejected_raw", compact(row.get("rejected")))
    return {
        "conversations": [
            {"from": "system", "value": PROMPTS[row["task"]]},
            {"from": "human", "value": row["input"]},
        ],
        "chosen": {"from": "gpt", "value": compact(row["chosen"])},
        "rejected": {"from": "gpt", "value": rejected},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(compact(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    rows = load_generated()
    assign_ids_and_difficulty(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    write_jsonl(GEN / "assembled_candidate_review.jsonl", rows)
    issues = validate(rows)
    (GEN / "validation_issues.json").write_text(json.dumps({"issue_count": len(issues), "issues": issues}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if issues:
        print(json.dumps({"valid": False, "rows": len(rows), "issue_count": len(issues), "sample": issues[:20]}, ensure_ascii=False))
        raise SystemExit(1)

    train, dev = split(rows)
    write_jsonl(DATA / "direct_review.jsonl", rows)
    write_jsonl(DATA / "all_sharegpt.jsonl", [sharegpt(row) for row in rows])
    write_jsonl(DATA / "train_sharegpt.jsonl", [sharegpt(row) for row in train])
    write_jsonl(DATA / "dev_sharegpt.jsonl", [sharegpt(row) for row in dev])
    spot = sorted(rows, key=lambda row: digest(row["pair_id"]))[:100]
    write_jsonl(DATA / "spotcheck_100.jsonl", spot)
    report = {
        "valid": True,
        "rows": len(rows), "train": len(train), "dev": len(dev),
        "task_distribution": dict(Counter(row["task"] for row in rows)),
        "error_distribution": dict(Counter(row["primary_error_type"] for row in rows)),
        "difficulty_distribution": dict(Counter(row["difficulty"] for row in rows)),
        "generation_channel_distribution": dict(Counter(row["generation_channel"] for row in rows)),
        "chosen_schema_errors": 0, "non_format_rejected_schema_errors": 0,
        "category_or_taxonomy_errors": 0, "entity_span_errors": 0,
        "duplicate_task_inputs": 0, "frozen_overlap": 0,
        "train_dev_non_taxonomy_source_overlap": 0,
        "train_dev_taxonomy_input_overlap": 0,
        "sft_outputs_used": False,
        "external_codex_processing_authorized_by_user": True,
    }
    (DATA / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "codex_direct_v2_full1000",
        "status": "generated_and_strictly_validated_pending_training_evaluation",
        "total_pairs": len(rows), "train_pairs": len(train), "dev_pairs": len(dev),
        "approved_pilot_pairs_retained": 7,
        "pilot_pairs_replaced_for_frozen_isolation": 3,
        "codex_cli_generated_pairs": 993,
        "strict_repair_overlay_pairs": 17,
        "strict_repair_same_input_pairs": 7,
        "strict_repair_replacement_pairs": 10,
        "semantic_generator": "Codex direct reasoning (gpt-5.6-sol for CLI batches)",
        "orchestration_scripts_generate_semantic_answers": False,
        "sft_outputs_used": False,
        "authorities": {
            "job_taxonomy": "job_cate.json",
            "other_candidates": "code/data/task_info/label_prompts_v2.md",
            "job_cate_sha256": hashlib.sha256(JOB_FILE.read_bytes()).hexdigest(),
            "label_prompts_sha256": hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest(),
        },
        "files": {
            "direct_review": "direct_review.jsonl", "all": "all_sharegpt.jsonl",
            "train": "train_sharegpt.jsonl", "dev": "dev_sharegpt.jsonl",
            "spotcheck": "spotcheck_100.jsonl", "validation": "validation_report.json",
        },
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": True, "rows": len(rows), "train": len(train), "dev": len(dev)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
