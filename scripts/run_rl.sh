#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-dpo}
CONFIG_PATH=${2:-configs/dpo.yaml}

if [[ "$MODE" == "dpo" ]]; then
  accelerate launch --config_file configs/accelerate.yaml src/rl/train_dpo.py --config "$CONFIG_PATH"
elif [[ "$MODE" == "ppo" ]]; then
  accelerate launch --config_file configs/accelerate.yaml src/rl/train_ppo.py --config "$CONFIG_PATH"
else
  echo "Unsupported mode: $MODE (expected: dpo|ppo)"
  exit 1
fi
