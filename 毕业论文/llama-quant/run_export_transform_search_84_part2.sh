#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SHARD_COUNT=4
export SHARD_INDEX=2
export LOG_DIR="${LOG_DIR:-logs_export_transform_search_84_part2_of_4}"
exec bash "${SCRIPT_DIR}/run_export_transform_search_84.sh" "$@"
