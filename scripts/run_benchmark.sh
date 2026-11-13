#!/usr/bin/env bash
# Option-GRPO on the twelve SDKs of UnseenSDK-Bench, three seeds each.
#   scripts/run_benchmark.sh [extra option-grpo arguments...]
set -euo pipefail

SDKS="${SDKS:-daft lancedb cuallee docling marker surya faststream tracecat mcp vortex libdebug reflexio}"
SEEDS="${SEEDS:-0 1 2}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for sdk in ${SDKS}; do
  for seed in ${SEEDS}; do
    "${HERE}/run_sdk.sh" "${sdk}" "${seed}" "$@"
  done
done
