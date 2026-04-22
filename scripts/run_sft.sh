#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/sft.yaml}
accelerate launch --config_file configs/accelerate.yaml src/sft/train_sft.py --config "$CONFIG_PATH"
