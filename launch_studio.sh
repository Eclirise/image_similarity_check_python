#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[INFO] 正在创建本地虚拟环境 .venv ..."
  python3 -m venv .venv
fi

PYTHON_EXE=".venv/bin/python"
REQ="requirements.txt"

if [[ ! -f "$REQ" ]]; then
  echo "[ERROR] 未找到 requirements.txt"
  exit 1
fi

set +e
"$PYTHON_EXE" -m pip install --upgrade pip >/dev/null 2>&1
for index in "https://pypi.tuna.tsinghua.edu.cn/simple" "https://pypi.org/simple"; do
  echo "[INFO] 尝试安装依赖源: $index"
  "$PYTHON_EXE" -m pip install -r "$REQ" -i "$index" --prefer-binary --disable-pip-version-check
  if [[ $? -eq 0 ]]; then
    echo "[INFO] 依赖安装成功: $index"
    set -e
    "$PYTHON_EXE" -m app.main
    exit 0
  fi
done
set -e

echo "[ERROR] 依赖安装失败，请检查网络或手动执行 pip 安装。"
exit 1
