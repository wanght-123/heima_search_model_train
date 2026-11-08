"""第 3 步：将四集合合并为 LLaMA-Factory ShareGPT 格式。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_finetune_course.data_processing import export_llamafactory_dataset  # noqa: E402
from job_finetune_course.project_config import project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default="code/data/datasets/candidates/gemini-v2-split-v2"
    )
    parser.add_argument("--output-dir", default="code/data/datasets/llamafactory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = export_llamafactory_dataset(
        project_path(args.input_dir),
        project_path(args.output_dir),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
