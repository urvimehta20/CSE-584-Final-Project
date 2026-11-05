#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
else
  if command -v python3.12 >/dev/null 2>&1; then
    BASE=python3.12
  elif command -v python3.11 >/dev/null 2>&1; then
    BASE=python3.11
  elif command -v python3.10 >/dev/null 2>&1; then
    BASE=python3.10
  else
    echo "Need Python 3.10+ (found: $(command -v python3 || true))"
    exit 1
  fi
  "$BASE" -m venv .venv
  PY=".venv/bin/python"
fi

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

RAW="data/raw/ClimRetrieve"
if [ ! -f "$RAW/Report-Level Dataset/ClimRetrieve_ReportLevel_V1.csv" ]; then
  mkdir -p data/raw
  rm -rf "$RAW"
  git clone --depth 1 https://github.com/tobischimanski/ClimRetrieve.git "$RAW"
fi

mkdir -p results/figures data/processed
export PYTHONPATH=src
"$PY" -u src/run_pilot.py
