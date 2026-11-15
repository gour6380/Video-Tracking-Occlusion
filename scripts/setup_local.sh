#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.13 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 13), "This project requires Python 3.13; preserve and rename an incompatible environment first"' 
uv pip sync requirements.txt --python .venv/bin/python --require-hashes
.venv/bin/python scripts/fetch_upstream.py
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name video-tracking-occlusion --display-name 'Video Tracking — Python 3.13'
