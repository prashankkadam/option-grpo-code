#!/usr/bin/env bash
# Build the execution sandbox image of an SDK.
#   scripts/build_sandbox.sh daft "daft==<version>" [extra requirements...]
set -euo pipefail

SDK="${1:?usage: build_sandbox.sh <sdk> <requirement> [extra requirements...]}"
REQUIREMENT="${2:?usage: build_sandbox.sh <sdk> <requirement> [extra requirements...]}"
shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker build \
  -f "${ROOT}/docker/sandbox.Dockerfile" \
  --build-arg SDK_REQUIREMENT="${REQUIREMENT}" \
  --build-arg EXTRA_REQUIREMENTS="$*" \
  -t "option-grpo-sandbox:${SDK}" \
  "${ROOT}/docker"

echo "built option-grpo-sandbox:${SDK}"
echo "use it with: -o execution.backend=docker -o execution.docker_image=option-grpo-sandbox:${SDK}"
