#!/usr/bin/env bash
# Assembles the submission payload on the training cluster:
#   1. model/      — MedNeXt-L k5 checkpoint_best.pth for all 5 folds + plans/dataset
#   2. vendor/     — the exact pure-Python nnU-Net v2 + MedNeXt source used for training
#
# Run this on the machine where $SCRATCH is mounted (e.g. the Compute Canada
# login node). Then transfer this directory to a machine with Docker and run
# ./do_build.sh and ./do_test_run.sh there.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Point these at your training environment (e.g. a Compute Canada scratch dir).
SCRATCH="${SCRATCH:-/scratch/<your-username>}"
VENV_SITE="${HOME}/nnunet_venv/lib/python3.11/site-packages"
MODEL_SRC="${SCRATCH}/nnUNet_results/Dataset800_ISLES2026/nnUNetTrainerMedNeXt_L_k5__nnUNetPlans__3d_fullres"
MEDNEXT_SRC="${SCRATCH}/MedNeXt/nnunet_mednext"

echo "==> Checking sources"
for p in "${MODEL_SRC}/plans.json" "${MODEL_SRC}/dataset.json" "${VENV_SITE}/nnunetv2" "${MEDNEXT_SRC}"; do
  [ -e "$p" ] || { echo "ERROR: missing $p" >&2; exit 1; }
done

echo "==> Packaging model (checkpoint_best.pth, 5 folds)"
mkdir -p model
cp -f "${MODEL_SRC}/plans.json" model/
cp -f "${MODEL_SRC}/dataset.json" model/
[ -f "${MODEL_SRC}/dataset_fingerprint.json" ] && cp -f "${MODEL_SRC}/dataset_fingerprint.json" model/ || true
for f in 0 1 2 3 4; do
  mkdir -p "model/fold_${f}"
  cp -f "${MODEL_SRC}/fold_${f}/checkpoint_best.pth" "model/fold_${f}/checkpoint_best.pth"
done

echo "==> Vendoring nnU-Net v2 + MedNeXt source (exact training code)"
rm -rf vendor
mkdir -p vendor
for pkg in nnunetv2 dynamic_network_architectures batchgenerators acvl_utils; do
  cp -r "${VENV_SITE}/${pkg}" "vendor/${pkg}"
  find "vendor/${pkg}" -type d -name '__pycache__' -prune -exec rm -rf {} +
  find "vendor/${pkg}" -type f -name '*.pyc' -delete
done

cp -r "${MEDNEXT_SRC}" "vendor/nnunet_mednext"
find "vendor/nnunet_mednext" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "vendor/nnunet_mednext" -type f -name '*.pyc' -delete

# MedNeXt-L uses gradient checkpointing; newer torch raises unless the
# use_reentrant argument is passed explicitly. Patch the vendored copy only.
MEDNEXT_PATCH="vendor/nnunet_mednext/network_architecture/mednextv1/MedNextV1.py"
if grep -q "use_reentrant" "$MEDNEXT_PATCH"; then
  echo "    (MedNextV1.py already patched)"
else
  sed -i 's/self\.dummy_tensor)/self.dummy_tensor, use_reentrant=False)/g' "$MEDNEXT_PATCH"
  echo "    Patched use_reentrant=False in MedNextV1.py"
fi

echo ""
echo "==> Done."
echo "    model/  : $(du -sh model | cut -f1)  (weights + plans)"
echo "    vendor/ : $(du -sh vendor | cut -f1)  (nnU-Net v2 + MedNeXt source)"
echo ""
echo "    Next: transfer this directory to a Docker host, then run"
echo "          ./do_build.sh && ./do_test_run.sh"
