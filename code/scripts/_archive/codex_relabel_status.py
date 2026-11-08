#!/usr/bin/env python3
"""Print status for Codex relabel shards."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path("code/data/datasets/llamafactory/codex_full_relabel")


def main() -> None:
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    done = {
        int(path.stem.split("-")[1])
        for path in (BASE / "relabel_shards").glob("shard-*.jsonl")
    }
    audit = {
        int(path.name.split("-")[1].split(".")[0])
        for path in (BASE / "audit_shards").glob("shard-*.review.jsonl")
    }
    all_ids = set(range(manifest["total_shards"]))
    complete = done & audit
    missing = sorted(all_ids - complete)
    prefix = 0
    while prefix in complete:
        prefix += 1
    print(json.dumps(
        {
            "total_shards": manifest["total_shards"],
            "complete_shards": len(complete),
            "complete_rows_estimate": sum(
                shard["count"]
                for shard in manifest["shards"]
                if int(shard["shard"].split("-")[1]) in complete
            ),
            "continuous_prefix_shards": prefix,
            "next_missing": missing[:20],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
