#!/usr/bin/env python3
"""Ask Codex to regenerate the small strict-validation repair overlay."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "code/data/datasets/dpo/codex_direct_v2_full1000/_generation"
REPAIRS = GEN / "repairs"
CODEX = "/opt/homebrew/bin/codex"

REQUESTS = (
    {
        "name": "entity_repairs",
        "seed": "entity_repairs.json",
        "prompts": ["prompts/entity_extraction.md"],
        "extra": (
            "这是严格边界修复：chosen中除position第二项规范职位、workMode枚举值外，"
            "每一个非空字符串（包括数组元素）都必须逐字、连续出现在input中；"
            "不得清洗标点、删除括号、拼接不连续片段或同义改写。"
            "同时保留primary_error_type指定的chosen/rejected偏好差。"
        ),
    },
    {
        "name": "job_replacements",
        "seed": "job_replacements.json",
        "prompts": ["prompts/job_name_classification.md", "job_cate.json"],
        "extra": (
            "这是职位分类替换：chosen的level1/level2/level3必须全部非空，且三元路径必须完整存在于job_cate.json；"
            "遇到同名三级职位时必须依据原文上下文选正确父路径。rejected按原primary_error_type构造。"
        ),
    },
    {
        "name": "mixed_replacements",
        "seed": "mixed_replacements.json",
        "prompts": [
            "prompts/entity_extraction.md", "prompts/work_type_classification.md",
            "prompts/workplace_classification.md", "prompts/taxonomy_leaf_to_parent.md",
            "job_cate.json",
        ],
        "extra": (
            "这是冻结集重合替换：逐条按seed中的task使用对应任务提示词。"
            "实体抽取chosen必须满足逐字连续原文边界规则；所有分类只能使用项目候选；"
            "taxonomy叶子的chosen必须是job_cate.json中的完整合法路径。"
        ),
    },
)


def valid_output(output_path: Path, seed_path: Path) -> bool:
    try:
        output = json.loads(output_path.read_text(encoding="utf-8"))
        seeds = json.loads(seed_path.read_text(encoding="utf-8"))
        records = output["records"]
        return (
            len(records) == len(seeds)
            and {row["seed_id"] for row in records} == {row["seed_id"] for row in seeds}
            and all(isinstance(row["chosen_json"], str) and isinstance(row["rejected_json"], str) for row in records)
        )
    except Exception:
        return False


def run(request: dict) -> dict:
    name = request["name"]
    seed_path = REPAIRS / request["seed"]
    output_path = REPAIRS / f"{name}_output.json"
    log_path = REPAIRS / f"{name}.log"
    if valid_output(output_path, seed_path):
        return {"request": name, "ok": True, "seconds": 0.0, "detail": "cached"}
    allowed = [f"repairs/{request['seed']}", "error_types.md", *request["prompts"]]
    prompt = (
        "你是招聘领域DPO数据严格复核模型。只允许读取：" + "、".join(allowed) + "。"
        "禁止读取任何SFT旧output、其他batch答案或最终DPO文件。"
        "对seed逐条独立生成新的chosen和rejected。chosen严格符合对应任务提示词、固定schema与项目候选；"
        "rejected由你按primary_error_type生成，是语义相关、真实模型可能犯的困难错误。"
        + request["extra"] +
        "每个seed恰好返回一条且seed_id原样保留。chosen_json是紧凑合法JSON字符串；"
        "rejected_json通常是紧凑合法JSON字符串，仅E11允许故意格式错误。review_reason用一句话解释偏好。"
        "职位树只做定向查询。只返回output schema规定的JSON。"
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
    return {"request": name, "ok": ok, "seconds": round(elapsed, 1), "detail": "ok" if ok else f"exit={process.returncode}"}


def main() -> None:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, request) for request in REQUESTS]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
