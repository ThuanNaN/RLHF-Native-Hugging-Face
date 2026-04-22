#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/rm.yaml}
accelerate launch --config_file configs/accelerate.yaml src/reward_model/train_rm.py --config "$CONFIG_PATH"
