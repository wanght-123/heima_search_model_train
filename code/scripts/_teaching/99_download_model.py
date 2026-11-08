"""从魔搭社区（ModelScope）下载 Qwen3 模型。

重要：本项目的代码和模型物料是分开存放的。代码在 `heima_search_model_train/`，
模型在 `环境和物料/models/` 目录下。这两个目录的相对位置可能因学生环境不同而变化，
所以**不要使用相对路径**，而是先找到自己机器上的 `环境和物料` 目录。

用法：
    # 第 1 步：先找到你机器上的「环境和物料」目录
    #   例如：/Users/你的用户名/Documents/黑马/.../环境和物料/
    #   或：  C:\\Users\\你的用户名\\Documents\\黑马\\...\\环境和物料\\

    # 第 2 步：下载模型到该目录下的 models/ 子目录
    # Mac/Linux
    uv run python code/scripts/99_download_model.py \\
        --cache-dir /你的路径/环境和物料/models

    # Windows
    uv run python code/scripts/99_download_model.py --cache-dir C:\\你的路径\\环境和物料\\models

    # 指定模型尺寸
    uv run python code/scripts/99_download_model.py --model Qwen/Qwen3-4B --cache-dir /你的路径/环境和物料/models

下载完成后，模型目录结构应为：
    <环境和物料>/models/Qwen3-1.7B/
    ├── config.json          # 模型结构声明
    ├── tokenizer_config.json
    ├── *.safetensors        # 模型权重文件（可能分片）
    └── ...
"""

import argparse
import sys
from pathlib import Path


def download_model(model_id: str, cache_dir: Path) -> None:
    """使用魔搭社区 SDK 下载模型。

    Args:
        model_id: 魔搭社区的模型 ID，如 "Qwen/Qwen3-4B"
        cache_dir: 模型缓存目录，模型会下载到 {cache_dir}/{model_id 最后一段}/
    """
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("错误：未安装 modelscope，请先执行：")
        print("  uv add modelscope")
        print("  或")
        print("  pip install modelscope")
        sys.exit(1)

    # 创建缓存目录
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在从魔搭社区下载模型：{model_id}")
    print(f"缓存目录：{cache_dir}")
    print("（首次下载需要一些时间，请耐心等待...）\n")

    # 调用魔搭社区 SDK 下载
    # snapshot_download 会返回本地模型路径
    local_model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        revision="master",  # 使用主分支
    )

    print(f"\n✓ 模型下载完成！")
    print(f"本地路径：{local_model_path}")
    print(f"\n下一步：")
    print(f"  1. 检查模型文件：ls -lh {local_model_path}")
    print(f"  2. 运行环境检查：uv run python code/scripts/00_check_env.py --model-path {local_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="魔搭社区模型 ID（默认：Qwen/Qwen3-1.7B）。可选：Qwen/Qwen3-4B, Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="【必填】你机器上「环境和物料/models/」的绝对路径。请先找到这个目录再运行本脚本。",
    )
    args = parser.parse_args()

    download_model(args.model, args.cache_dir)


if __name__ == "__main__":
    main()
