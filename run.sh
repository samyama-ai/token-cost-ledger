#!/usr/bin/env bash
# One command regenerates every number and figure from REAL data (FLORES-200).
#   ./run.sh                 # full devtest
#   ./run.sh --n-eval 200    # quick subset
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1090
source ~/projects/venv/bin/activate
cd src
python run_all.py "$@"
PAPER_DIR="${PAPER_FIG_DIR:-../../../graph_ws/samyama-research/papers/paper20-token-cost-ledger}"
if [ -d "$PAPER_DIR" ]; then
  python figures.py "$PAPER_DIR"
else
  echo "[run] paper dir not found ($PAPER_DIR); figures written to results/ only"
  python figures.py
fi
echo "[run] done. Results in results/, figures alongside."
