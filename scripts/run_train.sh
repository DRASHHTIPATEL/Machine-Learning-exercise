#!/usr/bin/env bash
set -euo pipefail

if [ -z "${VIRTUAL_ENV-}" ]; then
  echo "Activating .venv..."
  if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
  else
    echo "No virtualenv found. Creating one..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
  fi
fi

python src/train.py --config configs/config.yaml
