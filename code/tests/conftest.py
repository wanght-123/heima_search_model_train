"""让测试可以直接导入 code/src 中的演示模块。"""

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
