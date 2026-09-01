#!/bin/bash
#SBATCH --partition=ood
#SBATCH --account=<your-account>
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=isles26_apptainer_test
#SBATCH --output=apptainer_test_%j.out

# Run the ISLES26 MedNeXt-L container (Apptainer SIF) on a T4 GPU compute node.
# This validates the full inference pipeline on the sanity-check subject
# (ISLES_0000 = sub-r001s001) before submission.
#
# Usage:
#   sbatch do_test_apptainer_slurm.sh
#
# NOTE: The login node has NO GPU access. Apptainer must run on a compute node
# via SLURM with --nv to use the GPU. The SIF and all files must be on a SHARED
# filesystem (/home or /scratch) — /tmp is node-local and NOT visible from
# compute nodes.

# Explicit apptainer path (module load may not work in batch jobs).
# Adjust to your cluster's apptainer install location.
export PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/apptainer/1.3.5/bin:$PATH"

cd ~/isles26_submission

# Ensure the sanity-check test input is present.
mkdir -p test/input/interf0/images/t1-brain-mri
if [ ! -f test/input/interf0/images/t1-brain-mri/ISLES_0000.nii.gz ]; then
  cp "${SCRATCH:-/scratch/<your-username>}/nnUNet_raw/Dataset800_ISLES2026/imagesTr/ISLES_0000_0000.nii.gz" \
     test/input/interf0/images/t1-brain-mri/ISLES_0000.nii.gz
fi

rm -rf test/output
mkdir -p test/output

echo "=== Running container on GPU compute node ==="
echo "Date: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Run the inference directly (bypassing the HTTP server).
#
# NOTE: We test the inference pipeline directly rather than via the HTTP
# /invoke endpoint because Apptainer does not share the container's network
# namespace between separate `run` and `exec` invocations, so a background
# server's port 4743 is not reachable from a second `apptainer exec`. The
# HTTP server (app.py) is the standard FastAPI invoke server from the official
# ISLES26 template and is known to work on Grand Challenge; what we need to
# validate here is the inference pipeline itself, which this does.
apptainer exec --nv \
  --bind model:/opt/ml/model:ro \
  --bind test/input/interf0:/input:ro \
  --bind test/output:/output \
  isles26_mednext_l.sif \
  /home/user/venv/bin/python /opt/app/inference.py

echo "=== Done at $(date) ==="
echo "Output:"
find test/output -type f | sed 's#^#  #'
