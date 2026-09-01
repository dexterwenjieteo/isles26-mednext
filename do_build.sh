#!/usr/bin/env bash
# Builds the ISLES26 MedNeXt-L submission image.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26_mednext_l"

if [ ! -d "${SCRIPT_DIR}/vendor" ]; then
  echo "ERROR: vendor/ not found. Run ./package_model.sh first." >&2
  exit 1
fi

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1
