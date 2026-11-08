"""LLaMA-Factory 训练和模型导出命令封装。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


SUPPORTED_ACTIONS = ("train", "export", "api")


def build_command(action: str, config_path: Path) -> list[str]:
    """构造命令列表，不使用 shell 字符串，避免转义问题。"""

    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"不支持的 LLaMA-Factory 动作：{action}")
    return ["llamafactory-cli", action, str(config_path)]


def run_llamafactory(
    action: str,
    config_path: Path,
    *,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """执行 LLaMA-Factory；`dry_run` 只打印命令，适合课堂演示。"""

    if not config_path.exists():
        raise FileNotFoundError(f"配置不存在：{config_path}")
    command = build_command(action, config_path)
    print("即将执行：", " ".join(command))
    if dry_run:
        return command
    if shutil.which("llamafactory-cli") is None:
        raise RuntimeError(
            "未找到 llamafactory-cli，请使用 `uv run python ...` 启动本脚本"
        )
    env = os.environ.copy()
    env.update(extra_env or {})
    subprocess.run(command, check=True, env=env)
    return command
