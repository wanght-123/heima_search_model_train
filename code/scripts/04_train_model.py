"""第 4 步：使用 LLaMA-Factory 训练 Qwen3-4B LoRA。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_finetune_course.model_training import run_llamafactory  # noqa: E402
from job_finetune_course.project_config import project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code/configs/qwen3_4b_sft.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="只展示命令，不真正训练。"
    )
    args = parser.parse_args()
    run_llamafactory("train", project_path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
