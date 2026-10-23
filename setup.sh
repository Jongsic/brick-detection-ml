#!/usr/bin/env bash
# Creates .venv and work/nanodet (NanoDet at a pinned commit + patches/nanodet.patch).
set -euo pipefail
cd "$(dirname "$0")"
NANODET_COMMIT=be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0

[ -d .venv ] || "${PYTHON:-python3}" -m venv .venv
.venv/bin/pip install -q -r requirements.txt

if [ ! -d work/nanodet ]; then
  git init -q work/nanodet
  git -C work/nanodet fetch -q --depth 1 https://github.com/RangiLyu/nanodet.git "$NANODET_COMMIT"
  git -C work/nanodet checkout -q FETCH_HEAD
  git -C work/nanodet apply "$PWD/patches/nanodet.patch"
fi
echo "ready: .venv, work/nanodet"
