"""第 5 步：评估 SFT 模型的 JSON 和字段准确率。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_finetune_course.model_evaluation import evaluate_model  # noqa: E402
from job_finetune_course.project_config import project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="code/data/datasets/llamafactory/recruitment_sft_diagnostic.jsonl",
    )
    parser.add_argument("--model-path", default="model_files/Qwen3-4B")
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="可选的 LoRA Adapter 路径；不传时直接评测 model-path。",
    )
    parser.add_argument("--output-dir", default="code/outputs/evaluation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--batch-size", type=int, default=1, help="每批同时预测的样本数。"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--gold-as-prediction", action="store_true", help="不加载模型，只检查评估代码。"
    )
    args = parser.parse_args()

    metrics = evaluate_model(
        project_path(args.dataset),
        project_path(args.output_dir),
        model_path=project_path(args.model_path),
        adapter_path=(
            None
            if args.gold_as_prediction or args.adapter_path is None
            else project_path(args.adapter_path)
        ),
        limit=args.limit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        gold_as_prediction=args.gold_as_prediction,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
