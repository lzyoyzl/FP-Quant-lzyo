#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SHARD_COUNT=4
export SHARD_INDEX=1
export RUN_BASE="${RUN_BASE:-1}"
export LOG_DIR="${LOG_DIR:-lm_eval_logs_llama3_transform_search_84_part1_of_4}"
exec bash "${SCRIPT_DIR}/run_lm_eval_transform_search_84.sh" "$@"
