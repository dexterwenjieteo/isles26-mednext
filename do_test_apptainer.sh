#!/usr/bin/env bash
# Local end-to-end test of the ISLES26 MedNeXt-L container using Apptainer
# on nibi (Compute Canada). This validates the inference pipeline locally.
#
# NOTE: Apptainer produces a SIF image for TESTING ONLY. Grand Challenge
# requires a Docker image (docker save .tar.gz), which must be built on a
# machine with Docker. See README.md.
#
# Usage:
#   ./do_test_apptainer.sh          # build + run
#   ./do_test_apptainer.sh --build  # build only
#   ./do_test_apptainer.sh --run    # run only (assumes SIF exists)
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

SIF="isles26_mednext_l.sif"
INPUT_DIR="${SCRIPT_DIR}/test/input/interf0"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
MODEL_DIR="${SCRIPT_DIR}/model"

MODE="${1:-all}"

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

module load apptainer 2>/dev/null || true

if [ "$MODE" = "all" ] || [ "$MODE" = "--build" ]; then
  echo "==> Building Apptainer image (this pulls the ~3.6GB base image)"
  apptainer build --fakeroot --ignore-fakeroot-command "$SIF" Apptainer.def
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "--run" ]; then
  if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found. Run ./do_test_apptainer.sh --build first." >&2
    exit 1
  fi
  rm -rf "$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"

  GPU_ARGS=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_ARGS="--nv"
    echo "GPU detected — enabling --nv"
  else
    echo "No GPU — running on CPU (slow; 5-fold+TTA can take hours)"
  fi

  echo "==> Running container (this may take a long time on CPU)"
  apptainer run ${GPU_ARGS} \
    --bind "${MODEL_DIR}":/opt/ml/model:ro \
    --bind "${INPUT_DIR}":/input:ro \
    --bind "${OUTPUT_DIR}":/output \
    "$SIF"

  echo "==> Output written to ${OUTPUT_DIR}/"
  find "$OUTPUT_DIR" -type f | sed 's#^#  #'
fi
