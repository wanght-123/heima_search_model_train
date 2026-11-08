"""项目公共配置。

这个文件只放“多个模块都会用到”的常量，避免在各脚本里重复书写。
"""

from pathlib import Path


# project_config.py 位于 code/src/job_finetune_course/，向上三级就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[3]

TASK_NAMES = (
    "entity_extraction",
    "job_name_classification",
    "taxonomy_leaf_to_parent",
    "taxonomy_level2_to_children",
    "work_type_classification",
    "workplace_classification",
    "settlement_classification",
    "work_time_classification",
    "benefit_classification",
)

SPLIT_NAMES = ("train", "dev", "diagnostic", "frozen_test")


def project_path(path: str | Path) -> Path:
    """将相对路径转换成相对项目根目录的绝对路径。"""

    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
