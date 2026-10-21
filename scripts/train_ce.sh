#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/ce_tiny_s2.yaml}"
shift "$(( $# > 0 ? 1 : 0 ))"
python -m src.training.train_ce --config "$CONFIG_PATH" "$@"
