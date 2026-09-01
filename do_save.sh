#!/usr/bin/env bash
# Saves the algorithm image and model as tarballs for upload to Grand Challenge.
# Run this AFTER ./do_test_run.sh confirms the container works.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26_mednext_l"
export DOCKER_CLI_HINTS=false

log() { printf "> %s\n" "$1"; }

log "Building image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"

build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG" )
if [ -z "$build_timestamp" ]; then
  echo "Error: failed to inspect image $DOCKER_IMAGE_TAG" >&2
  exit 1
fi
formatted=$( echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g' )
image_tarball="${DOCKER_IMAGE_TAG}_${formatted}.tar.gz"

log "Saving image tarball (this can take a while)"
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "${SCRIPT_DIR}/${image_tarball}"
log "Saved: ${image_tarball}"

log "Packing model tarball"
tar -czf "${SCRIPT_DIR}/model.tar.gz" -C "${SCRIPT_DIR}/model" .
log "Saved: model.tar.gz"

log "Upload ${image_tarball} as the algorithm image, and model.tar.gz as a separate Model."
