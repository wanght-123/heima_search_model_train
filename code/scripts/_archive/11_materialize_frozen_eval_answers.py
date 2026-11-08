#!/usr/bin/env python3
"""Create an immutable answer copy for the frozen quick evaluation set.

The source is never modified. Prompt text is copied byte-for-byte; this program
only materializes the separately reviewed assistant answer dataset and records
integrity metadata.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/datasets/llamafactory"
SOURCE = DATA / "recruitment_sft_frozen_test_quick_30pct.jsonl"
TARGET = DATA / "recruitment_sft_frozen_test_quick_30pct_model_answers_v1.jsonl"
MANIFEST = DATA / "recruitment_sft_frozen_test_quick_30pct_model_answers_v1.manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_hash_before = sha256(SOURCE)
    task_counts: Counter[str] = Counter()
    prompt_hashes: Counter[str] = Counter()
    rows = 0

    with SOURCE.open(encoding="utf-8") as reader, TARGET.open("w", encoding="utf-8") as writer:
        for line_no, line in enumerate(reader, 1):
            record = json.loads(line)
            conversations = record["conversations"]
            if [turn["from"] for turn in conversations] != ["system", "human", "gpt"]:
                raise ValueError(f"line {line_no}: unexpected conversation roles")
            json.loads(conversations[2]["value"])
            system = conversations[0]["value"]
            task_counts[system.split("\n", 1)[0]] += 1
            prompt_hashes[hashlib.sha256(system.encode("utf-8")).hexdigest()] += 1
            writer.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")
            rows += 1

    source_hash_after = sha256(SOURCE)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source dataset changed while materializing answers")

    manifest = {
        "source_file": SOURCE.name,
        "source_sha256": source_hash_before,
        "source_unchanged_after_write": True,
        "output_file": TARGET.name,
        "output_sha256": sha256(TARGET),
        "records": rows,
        "prompt_preservation": "system and human turns copied byte-for-byte",
        "assistant_answer_format": "JSON object per original task prompt",
        "task_counts": task_counts,
        "system_prompt_sha256_counts": prompt_hashes,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
