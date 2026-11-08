"""第 7 步：启动 LLaMA-Factory OpenAI 兼容 API。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_finetune_course.model_deployment import start_api  # noqa: E402
from job_finetune_course.project_config import project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code/configs/qwen3_4b_api.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="qwen3-4b-recruitment")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    start_api(
        project_path(args.config),
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
