#!/usr/bin/env bash
# 生成 Linux GPU 服务器包；可选包含 model_files 下的基座模型。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/dist"
INCLUDE_MODELS=false

usage() {
  echo "用法：bash code/scripts/09_pack_for_server.sh [--include-models]"
  echo
  echo "默认只打包代码、配置、数据、文档和环境锁文件。"
  echo "  --include-models  额外打包项目 model_files/ 目录中的全部文件"
  echo "  -h, --help        显示帮助"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-models)
      INCLUDE_MODELS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PACKAGE_SUFFIX=""
if [[ "${INCLUDE_MODELS}" == true ]]; then
  if [[ ! -d "${PROJECT_ROOT}/model_files" ]]; then
    echo "指定了 --include-models，但目录不存在：${PROJECT_ROOT}/model_files" >&2
    exit 1
  fi
  PACKAGE_SUFFIX="-with-models"
fi

OUTPUT_FILE="${OUTPUT_DIR}/recruitment-sft${PACKAGE_SUFFIX}-$(date +%Y%m%d).tar.gz"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

TAR_INPUTS=(
  code/scripts
  code/src
  code/tests
  code/configs
  code/data/gemini_labeled_v2/tasks_fixed
  code/data/datasets/llamafactory
  README.md
  code/README.md
  doc
  pyproject.toml
  uv.lock
)

# 切分后的数据集（如果存在就打包，否则警告）
SPLIT_DIR="code/data/datasets/candidates/gemini-v2-split-v2"
if [[ -d "${SPLIT_DIR}" ]]; then
  TAR_INPUTS+=("${SPLIT_DIR}")
else
  echo "⚠ 警告：切分数据集目录不存在：${SPLIT_DIR}" >&2
  echo "  如需打包切分数据，请先运行：" >&2
  echo "  python3 code/scripts/02_split_dataset.py --output-dir ${SPLIT_DIR}" >&2
fi

if [[ "${INCLUDE_MODELS}" == true ]]; then
  TAR_INPUTS+=(model_files)
fi

tar -czf "${OUTPUT_FILE}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  "${TAR_INPUTS[@]}"

tar -tzf "${OUTPUT_FILE}" >/dev/null
echo "服务器包：${OUTPUT_FILE}"
du -sh "${OUTPUT_FILE}"
if [[ "${INCLUDE_MODELS}" == true ]]; then
  echo "打包结果已包含：${PROJECT_ROOT}/model_files"
else
  echo "打包结果不包含基座模型；如需包含，请增加参数 --include-models。"
fi
