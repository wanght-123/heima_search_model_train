"""第 8 步：测试 API 的并发、延迟、吞吐和稳定性。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_finetune_course.model_deployment import benchmark, save_benchmark_report  # noqa: E402
from job_finetune_course.project_config import project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="code/data/datasets/llamafactory/recruitment_sft_diagnostic.jsonl",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-4b-recruitment")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default="code/outputs/service_benchmark.json")
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        raise ValueError("--requests 和 --concurrency 必须大于 0")
    summary, results = benchmark(
        project_path(args.dataset),
        url=args.url,
        model=args.model,
        request_count=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        api_key=args.api_key,
    )
    save_benchmark_report(project_path(args.output), summary, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["success_rate"] < 1.0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
