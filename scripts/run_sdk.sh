#!/usr/bin/env bash
# Full pipeline for one SDK: Stage 0, three training phases, evaluation.
#   scripts/run_sdk.sh daft [seed] [extra option-grpo arguments...]
#
# Environment:
#   CONFIGS   space-separated config files   (default: configs/default.yaml)
#   LAUNCHER  command prefix for training    (default: plain python; e.g.
#             "accelerate launch --config_file configs/accelerate/zero3.yaml -m option_grpo.cli")
set -euo pipefail

SDK="${1:?usage: run_sdk.sh <sdk> [seed] [extra args...]}"
SEED="${2:-0}"
shift $(( $# > 1 ? 2 : 1 ))
CONFIGS="${CONFIGS:-configs/default.yaml}"
LAUNCHER="${LAUNCHER:-python -m option_grpo.cli}"
RUN="runs/${SDK}/seed${SEED}"

CONFIG_ARGS=()
for config in ${CONFIGS}; do CONFIG_ARGS+=(-c "${config}"); done

if [ ! -f "artifacts/${SDK}/tasks.jsonl" ]; then
  python -m option_grpo.cli prepare "${CONFIG_ARGS[@]}" --sdk "${SDK}" "$@"
fi
${LAUNCHER} train "${CONFIG_ARGS[@]}" --sdk "${SDK}" --output-dir "${RUN}" -o "seed=${SEED}" --no-eval "$@"
python -m option_grpo.cli evaluate "${CONFIG_ARGS[@]}" --sdk "${SDK}" --output-dir "${RUN}" \
  --checkpoint "${RUN}/checkpoints/final" -o "seed=${SEED}" "$@"
python -m option_grpo.cli analyze odl "${RUN}/eval"
