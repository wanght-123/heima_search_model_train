#!/usr/bin/env python3
"""Prepare narrowly scoped Codex repair/replacement requests.

The script only packages task inputs, prior Codex answers, validation feedback,
and fixed error targets. It never reads or exports any SFT ``output`` field.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "code/data/datasets/dpo/codex_direct_v2_full1000/_generation"
REPAIRS = GEN / "repairs"
ASSEMBLED = GEN / "assembled_candidate_review.jsonl"
SFT_ROOT = ROOT / "code/data/gemini_labeled_v2/tasks_fixed"
FROZEN = ROOT / "code/data/datasets/llamafactory/recruitment_sft_frozen_test.jsonl"

ENTITY_REPAIR_IDS = {
    "DIRECT_NEW_0107": "chosen preferredQualifications contains a normalized paraphrase; every non-empty span must be an exact input substring",
    "DIRECT_NEW_0065": "chosen location combines non-contiguous text; every non-empty span must be an exact input substring",
    "DIRECT_NEW_0020": "chosen responsibilities contains a shortened non-verbatim span",
    "DIRECT_NEW_0086": "chosen responsibilities removes a parenthetical segment and is not verbatim",
    "DIRECT_NEW_0157": "chosen responsibilities contains a non-contiguous paraphrase",
    "DIRECT_NEW_0095": "chosen responsibilities removes a parenthetical segment and is not verbatim",
    "DIRECT_NEW_0036": "chosen requirements paraphrases the ISO/MLPS experience instead of copying an exact span",
}

# Preserve each invalid pair's task and error type while moving it to a clear,
# previously unused input. The replacements are intentionally selected by source
# identity only; their historical SFT labels are neither accessed nor packaged.
JOB_REPLACEMENTS = {
    "DIRECT_NEW_0485": "job_031701",  # logistics specialist/assistant
    "DIRECT_NEW_0471": "job_028904",  # collection specialist
    "DIRECT_NEW_0680": "sq_000595",   # Korean translator, JSON-format negative
    "DIRECT_NEW_0470": "sq_002401",   # film distribution
    "DIRECT_NEW_0676": "job_023389",  # clinical project manager: context resolves duplicate leaf
    "DIRECT_NEW_0505": "job_000274",  # HPC engineer: AI context resolves duplicate leaf
}

MIXED_REPLACEMENTS = {
    "CODEX_DPO_P10_001": ("entity_extraction", "job_007708"),
    "CODEX_DPO_P10_006": ("work_type_classification", "job_000229"),
    "CODEX_DPO_P10_007": ("workplace_classification", "job_000002"),
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def load_input(task: str, source_id: str) -> str:
    path = SFT_ROOT / task / "train.jsonl"
    for row in read_jsonl(path):
        if str(row.get("source_id", "")) == source_id:
            # Deliberately access only input/source_id. Never access row["output"].
            return str(row["input"])
    raise KeyError(f"missing source {task}/{source_id}")


def compact_record(row: dict) -> dict:
    keys = (
        "seed_id", "task", "source_type", "source_id", "input",
        "primary_error_type", "target_field",
    )
    return {key: row[key] for key in keys}


def main() -> None:
    REPAIRS.mkdir(parents=True, exist_ok=True)
    current = {row["seed_id"]: row for row in read_jsonl(ASSEMBLED)}
    used_inputs = {row["input"] for row in current.values()}
    frozen = set()
    frozen_norm = set()
    for row in read_jsonl(FROZEN):
        value = next((str(m.get("value", "")) for m in row.get("conversations", []) if m.get("from") == "human"), "")
        frozen.add(value)
        frozen_norm.add(norm(value))

    entity_repairs = []
    for seed_id, issue in ENTITY_REPAIR_IDS.items():
        old = current[seed_id]
        entity_repairs.append({
            **compact_record(old),
            "repair_mode": "regenerate_same_input",
            "validation_feedback": issue,
            "previous_chosen_json": json.dumps(old["chosen"], ensure_ascii=False, separators=(",", ":")),
            "previous_rejected_json": json.dumps(old["rejected"], ensure_ascii=False, separators=(",", ":")),
        })

    job_replacements = []
    for seed_id, source_id in JOB_REPLACEMENTS.items():
        old = current[seed_id]
        value = load_input("job_name_classification", source_id)
        job_replacements.append({
            **compact_record(old),
            "source_id": source_id,
            "input": value,
            "repair_mode": "replace_ambiguous_or_unsupported_input",
            "validation_feedback": "previous Codex chosen path was empty; generate a complete legal job_cate.json path for this clearer input",
        })

    mixed_replacements = []
    for seed_id, (task, source_id) in MIXED_REPLACEMENTS.items():
        old = current[seed_id]
        value = load_input(task, source_id)
        mixed_replacements.append({
            **compact_record(old),
            "task": task,
            "source_id": source_id,
            "input": value,
            "repair_mode": "replace_frozen_overlap",
            "validation_feedback": "previous input overlaps the frozen test set; generate a fresh pair for this unused input",
        })

    tax_old = current["DIRECT_NEW_0965"]
    mixed_replacements.append({
        **compact_record(tax_old),
        "source_id": "job_cate.json",
        "input": "三级职位：Java",
        "repair_mode": "replace_frozen_overlap",
        "validation_feedback": "previous taxonomy input overlaps the frozen test set; use this unused unambiguous leaf",
    })

    replacement_inputs = [row["input"] for row in job_replacements + mixed_replacements]
    if len(replacement_inputs) != len(set(replacement_inputs)):
        raise ValueError("duplicate replacement inputs")
    for value in replacement_inputs:
        if value in used_inputs or value in frozen or norm(value) in frozen_norm:
            raise ValueError(f"replacement input is used or frozen: {value[:80]!r}")

    files = {
        "entity_repairs.json": entity_repairs,
        "job_replacements.json": job_replacements,
        "mixed_replacements.json": mixed_replacements,
    }
    for name, rows in files.items():
        (REPAIRS / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "repair_count": len(entity_repairs),
        "replacement_count": len(job_replacements) + len(mixed_replacements),
        "total_overlay_count": sum(len(rows) for rows in files.values()),
        "sft_outputs_used": False,
        "files": {name: len(rows) for name, rows in files.items()},
        "overlay_seed_ids": [row["seed_id"] for rows in files.values() for row in rows],
    }
    (REPAIRS / "repair_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
