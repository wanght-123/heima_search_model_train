"""第 2 步：按 source_id 切分四个数据集，防止同源样本泄漏。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_finetune_course.data_processing import load_task_rows  # noqa: E402
from job_finetune_course.dataset_split import save_splits, split_by_source  # noqa: E402
from job_finetune_course.project_config import project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default="code/data/gemini_labeled_v2/tasks_fixed"
    )
    parser.add_argument(
        "--output-dir", required=True, help="请使用新的版本目录，脚本不会覆盖旧数据。"
    )
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    rows = load_task_rows(project_path(args.input_dir))
    splits = split_by_source(rows, seed=args.seed)
    manifest = save_splits(project_path(args.output_dir), splits, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
