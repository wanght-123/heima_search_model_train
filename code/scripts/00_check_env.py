"""检查 Qwen3-4B + LLaMA-Factory SFT 的最小运行环境。"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_MODULES = ("torch", "transformers", "peft", "llamafactory")


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def nvidia_smi() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 3:
            rows.append(
                {"name": values[0], "memory_mib": values[1], "driver": values[2]}
            )
    return rows


def model_metadata(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {"exists": model_path.exists(), "config_exists": False}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "config_exists": True,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "torch_dtype": config.get("torch_dtype"),
        "safetensors": len(list(model_path.glob("*.safetensors"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("model_files/Qwen3-4B"))
    parser.add_argument(
        "--require-cuda", action="store_true", help="CUDA 不可用时返回失败。"
    )
    args = parser.parse_args()

    versions = {name: module_version(name) for name in REQUIRED_MODULES}
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        cuda_devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_gib": round(
                    torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        cuda_available = False
        cuda_version = None
        cuda_devices = []

    model = model_metadata(args.model_path.resolve())
    python_ok = sys.version_info[:2] == (3, 11)
    modules_ok = all(version is not None for version in versions.values())
    model_ok = bool(model.get("config_exists") and model.get("safetensors"))
    ready = (
        python_ok
        and modules_ok
        and model_ok
        and (cuda_available or not args.require_cuda)
    )
    report = {
        "ready": ready,
        "python": platform.python_version(),
        "python_3_11": python_ok,
        "platform": platform.platform(),
        "modules": versions,
        "model_path": str(args.model_path.resolve()),
        "model": model,
        "cuda": {
            "required": args.require_cuda,
            "available": cuda_available,
            "runtime_version": cuda_version,
            "torch_devices": cuda_devices,
            "nvidia_smi": nvidia_smi(),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
