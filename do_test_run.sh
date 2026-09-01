#!/usr/bin/env bash
# Builds the image and exercises it end-to-end against a local test case,
# mimicking the Grand Challenge invoke API.
#
#   test/input/interf0/images/t1-brain-mri/  ->  /input/images/t1-brain-mri
#   test/input/interf0/stroke-metadata.json  ->  /input/stroke-metadata.json
#   model/                                    ->  /opt/ml/model
#   test/output/                              ->  /output
#
# Run:  ./do_test_run.sh
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26_mednext_l"
CONTAINER_NAME="isles26_mednext_l_test"
PORT=4743

INPUT_DIR="${SCRIPT_DIR}/test/input/interf0"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
MODEL_DIR="${SCRIPT_DIR}/model"

# CPU 5-fold ensemble can take a long time; GPU is strongly recommended.
INVOKE_TIMEOUT_SECONDS="${INVOKE_TIMEOUT_SECONDS:-3600}"

export DOCKER_CLI_HINTS=false

if [ ! -d "${MODEL_DIR}/fold_0" ]; then
  echo "ERROR: model/ not found. Run ./package_model.sh first." >&2
  exit 1
fi
if ! ls "${INPUT_DIR}/images/t1-brain-mri/"*.nii.gz >/dev/null 2>&1 && \
   ! ls "${INPUT_DIR}/images/t1-brain-mri/"*.nii >/dev/null 2>&1 && \
   ! ls "${INPUT_DIR}/images/t1-brain-mri/"*.mha >/dev/null 2>&1; then
  echo "ERROR: no test image in ${INPUT_DIR}/images/t1-brain-mri/" >&2
  exit 1
fi

# Build.
docker build --platform=linux/amd64 --tag "$DOCKER_IMAGE_TAG" "$SCRIPT_DIR"

# Clean state.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Permissions: the container runs as non-root 'user'.
chmod -R o+rX "$MODEL_DIR" "$INPUT_DIR" 2>/dev/null || true
chmod o+rwX "$OUTPUT_DIR"

GPU_ARGS=""
# Only enable the GPU if there is an ACTUAL NVIDIA GPU present. Checking for the
# nvidia runtime alone is not enough — the nvidia-container-toolkit may be
# installed (so docker info lists the runtime) even when the machine has no
# NVIDIA GPU (e.g. Intel Arc, or WSL without a GPU). In that case `--gpus all`
# fails with "no adapters were found". nvidia-smi only works with a real GPU.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU_ARGS="--gpus all"
  echo "NVIDIA GPU detected — enabling GPU"
else
  echo "No NVIDIA GPU detected — running on CPU (slow; GPU recommended)"
fi

docker run --detach --name "$CONTAINER_NAME" \
  -p "${PORT}:4743" \
  ${GPU_ARGS:+$GPU_ARGS} \
  --platform=linux/amd64 \
  --volume "${MODEL_DIR}":/opt/ml/model:ro \
  --volume "${INPUT_DIR}":/input:ro \
  --volume "${OUTPUT_DIR}":/output \
  "$DOCKER_IMAGE_TAG"

cleanup() {
  # Only remove the container on SUCCESS. On failure we leave it running so the
  # user can inspect logs with: docker logs -f isles26_mednext_l_test
  if [ "${KEEP_ON_FAILURE:-0}" = "1" ]; then
    return
  fi
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Waiting for /health ..."
healthy=0
for i in $(seq 1 60); do
  code=$( curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://localhost:${PORT}/health" || true )
  if [ "$code" = "200" ]; then healthy=1; echo "Healthy (attempt $i)"; break; fi
  # Show recent container logs so we can see progress/errors while waiting.
  if [ $((i % 3)) -eq 0 ]; then
    echo "--- container logs (attempt $i) ---"
    docker logs --tail 15 "$CONTAINER_NAME" 2>&1 || true
  fi
  sleep 10
done
if [ "$healthy" != "1" ]; then
  echo "ERROR: /health never returned 200. Full logs:" >&2
  docker logs "$CONTAINER_NAME" >&2 || true
  echo "Container left running for inspection: docker logs -f $CONTAINER_NAME" >&2
  KEEP_ON_FAILURE=1
  exit 1
fi

echo "Calling /invoke (timeout ${INVOKE_TIMEOUT_SECONDS}s) ..."
code=$( curl -s -o /dev/null -w "%{http_code}" -X POST --max-time "$INVOKE_TIMEOUT_SECONDS" "http://localhost:${PORT}/invoke" || true )
if [ "$code" != "201" ]; then
  echo "ERROR: /invoke returned $code (expected 201). Logs:" >&2
  docker logs "$CONTAINER_NAME" >&2 || true
  exit 1
fi

echo "Invoke completed (201)."
docker logs "$CONTAINER_NAME" || true
echo "Output written to: ${OUTPUT_DIR}/"
find "$OUTPUT_DIR" -type f | sed 's#^#  #'
