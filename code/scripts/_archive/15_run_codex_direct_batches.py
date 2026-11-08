#!/usr/bin/env python3
"""Run Codex as the semantic generator for prepared DPO batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "code/data/datasets/dpo/codex_direct_v2_full1000/_generation"
CODEX = "/opt/homebrew/bin/codex"


def valid_output(path: Path, seed_path: Path) -> bool:
    try:
        output = json.loads(path.read_text(encoding="utf-8"))
        seeds = json.loads(seed_path.read_text(encoding="utf-8"))
        records = output["records"]
        return (
            len(records) == len(seeds)
            and {row["seed_id"] for row in records} == {row["seed_id"] for row in seeds}
            and all(isinstance(row["chosen_json"], str) and isinstance(row["rejected_json"], str) for row in records)
        )
    except Exception:
        return False


def run_batch(batch: dict, attempt: int = 1) -> tuple[str, bool, float, str]:
    batch_id = batch["batch_id"]
    task = batch["task"]
    seed_path = GEN / "batches" / batch["seed_file"]
    output_path = GEN / "outputs" / f"{batch_id}.json"
    log_path = GEN / "logs" / f"{batch_id}.attempt{attempt}.log"
    if valid_output(output_path, seed_path):
        return batch_id, True, 0.0, "cached"

    allowed = [f"prompts/{task}.md", "error_types.md", f"batches/{batch['seed_file']}"]
    if task in {"job_name_classification", "taxonomy_leaf_to_parent", "taxonomy_level2_to_children"}:
        allowed.append("job_cate.json")
    allowed_text = "、".join(allowed)
    prompt = (
        "你是招聘领域DPO数据标注模型。只允许读取：" + allowed_text + "。"
        "禁止读取任何SFT旧output、历史DPO答案或其他答案文件。"
        "对batch中每个seed逐条独立推理：chosen必须完全符合任务提示词、固定schema和项目候选；"
        "rejected必须由你按primary_error_type生成，是真实模型可能犯的困难错误，且与chosen存在明确偏好差。"
        "每个seed恰好返回一条，seed_id原样保留，不得跳过或新增。"
        "chosen_json是chosen对象的紧凑合法JSON字符串；rejected_json通常是rejected对象的紧凑合法JSON字符串，"
        "只有E11可输出故意非纯JSON或错schema文本。review_reason用一句话说明偏好原因。"
        "对职位树请用命令定向查询job_cate.json，不要一次打印整个文件。"
        "只返回output schema规定的JSON。"
    )
    command = [
        CODEX, "exec", "--ephemeral", "--ignore-user-config",
        "-m", "gpt-5.6-sol", "-c", 'model_reasoning_effort="medium"',
        "-s", "read-only", "--skip-git-repo-check", "-C", str(GEN),
        "--output-schema", str(GEN / "output_schema.json"),
        "-o", str(output_path), prompt,
    ]
    started = time.time()
    process = subprocess.run(command, text=True, capture_output=True, timeout=900)
    elapsed = time.time() - started
    log_path.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
    ok = process.returncode == 0 and valid_output(output_path, seed_path)
    detail = "ok" if ok else f"exit={process.returncode}"
    return batch_id, ok, elapsed, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads((GEN / "seed_manifest.json").read_text(encoding="utf-8"))
    batches = [row for row in manifest["batches"] if not args.only or args.only in row["batch_id"]]
    if args.limit:
        batches = batches[:args.limit]
    pending = batches
    for attempt in range(1, args.retries + 2):
        if not pending:
            break
        print(json.dumps({"attempt": attempt, "pending": len(pending)}, ensure_ascii=False), flush=True)
        failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(run_batch, batch, attempt): batch for batch in pending}
            for future in concurrent.futures.as_completed(future_map):
                batch = future_map[future]
                try:
                    batch_id, ok, elapsed, detail = future.result()
                except Exception as exc:
                    batch_id, ok, elapsed, detail = batch["batch_id"], False, 0.0, repr(exc)
                print(json.dumps({"batch": batch_id, "ok": ok, "seconds": round(elapsed, 1), "detail": detail}, ensure_ascii=False), flush=True)
                if not ok:
                    failed.append(batch)
        pending = failed
    completed = sum(valid_output(GEN / "outputs" / f"{row['batch_id']}.json", GEN / "batches" / row["seed_file"]) for row in batches)
    print(json.dumps({"selected": len(batches), "completed": completed, "failed": [row["batch_id"] for row in pending]}, ensure_ascii=False), flush=True)
    if pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
