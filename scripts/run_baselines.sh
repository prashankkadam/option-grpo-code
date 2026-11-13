#!/usr/bin/env bash
# Baselines and ablations for one SDK (same base model, same tasks, same reward).
#   scripts/run_baselines.sh daft [seed]
set -euo pipefail

SDK="${1:?usage: run_baselines.sh <sdk> [seed]}"
SEED="${2:-0}"
CONFIGS="${CONFIGS:-configs/default.yaml}"
LAUNCHER="${LAUNCHER:-python -m option_grpo.cli}"
BASE="runs/${SDK}/seed${SEED}"

CONFIG_ARGS=()
for config in ${CONFIGS}; do CONFIG_ARGS+=(-c "${config}"); done

# Certified options of the Option-GRPO run, used to measure idiomaticity for every method.
OPTIONS="${BASE}/checkpoints/final/options.json"
OPTION_ARGS=()
if [ -f "${OPTIONS}" ]; then OPTION_ARGS=(--options "${OPTIONS}"); fi

# Training-free baselines.
for method in frozen rag docprompting; do
  python -m option_grpo.cli baseline "${CONFIG_ARGS[@]}" --sdk "${SDK}" --method "${method}" \
    --output-dir "${BASE}" -o "seed=${SEED}" "${OPTION_ARGS[@]}"
done

# Hierarchy ablations.
for ablation in flat_grpo curriculum_only options_only; do
  RUN="${BASE}-${ablation}"
  ${LAUNCHER} train "${CONFIG_ARGS[@]}" -c "configs/ablations/${ablation}.yaml" --sdk "${SDK}" \
    --output-dir "${RUN}" -o "seed=${SEED}" --no-eval
  FLAT=()
  if [ "${ablation}" != "options_only" ]; then FLAT=(--flat "${OPTION_ARGS[@]}"); fi
  python -m option_grpo.cli evaluate "${CONFIG_ARGS[@]}" -c "configs/ablations/${ablation}.yaml" --sdk "${SDK}" \
    --output-dir "${RUN}" --checkpoint "${RUN}/checkpoints/final" -o "seed=${SEED}" "${FLAT[@]}"
done

# Complementarity with retrieval at inference time.
for retrieval in rag docprompting; do
  python -m option_grpo.cli evaluate "${CONFIG_ARGS[@]}" --sdk "${SDK}" --output-dir "${BASE}" \
    --checkpoint "${BASE}/checkpoints/final" --retrieval "${retrieval}" --name "eval-${retrieval}" -o "seed=${SEED}"
done
