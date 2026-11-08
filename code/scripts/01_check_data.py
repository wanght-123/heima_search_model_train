"""第 1 步：读取 tasks_fixed，检查基础格式并统计九任务数量。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_finetune_course.data_processing import build_data_report, load_task_rows  # noqa: E402
from job_finetune_course.project_config import project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default="code/data/gemini_labeled_v2/tasks_fixed"
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="每个任务只检查前 N 条，0 表示全量。",
    )
    parser.add_argument("--output", default="code/outputs/data_check_report.json")
    args = parser.parse_args()

    rows = load_task_rows(
        project_path(args.input_dir), limit_per_task=args.limit_per_task
    )
    report = build_data_report(rows)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
