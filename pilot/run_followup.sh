#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
else
  echo "Create the pilot venv first: bash run_all.sh"
  exit 1
fi

"$PY" -m pip install -q -r requirements.txt

if [ ! -f data/processed/chunks.parquet ] || [ ! -f data/processed/embeddings/chunk_embeddings.npy ]; then
  echo "Pilot tables/embeddings missing; running the original pilot first."
  bash run_all.sh
fi

export PYTHONPATH=src
"$PY" -u src/setting_g_stats.py
"$PY" -u src/run_followup.py
"$PY" -u src/run_e_human.py || true
