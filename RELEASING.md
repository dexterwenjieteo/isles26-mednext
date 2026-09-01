# How to publish a new model release

This file is for the maintainer. The 5-fold ensemble is ~2.4 GB and is delivered as **per-fold GitHub Release assets** rather than LFS-tracked files.

> **Why per-fold tarballs?** GitHub has two hard limits:
> - Regular repo files: **100 MB** per file (rejected if larger)
> - Release assets: **2 GiB** per file
>
> A single `model.tar.gz` is 2.19 GiB — just over the 2 GiB Release limit. Splitting into per-fold tarballs (449 MB each) fits comfortably and lets users download only the folds they need.

## Step 1 — assemble the tarballs

```bash
cd /path/to/build/dir

# Per-fold checkpoints (449 MB each)
for f in 0 1 2 3 4; do
  tar -czf fold_$f.tar.gz -C model fold_$f
done

# Metadata (plans.json, dataset.json, dataset_fingerprint.json)
tar -czf model_metadata.tar.gz -C model plans.json dataset.json dataset_fingerprint.json

# Record checksums
sha256sum fold_*.tar.gz model_metadata.tar.gz
```

## Step 2 — tag the repo

```bash
git tag -a v1.0.0 -m "ISLES 2026 — MedNeXt-L k=5, 5-fold ensemble, Dice 0.6629"
git push origin v1.0.0
```

## Step 3 — attach to the GitHub release

1. Go to **Releases → Draft a new release**.
2. Choose tag `v1.0.0`.
3. Title: `ISLES 2026 — MedNeXt-L k=5, 5-fold ensemble`
4. Paste the SHA-256 checksums of all 6 tarballs in the description.
5. Drag-and-drop the 6 tarballs into the "Attach binaries" area.
6. Publish.

## Step 4 — verify

After publishing, run the following on a fresh checkout to confirm the workflow is intact:

```bash
# Download all 6 assets
for f in 0 1 2 3 4; do
  wget https://github.com/<owner>/<repo>/releases/download/v1.0.0/fold_$f.tar.gz
done
wget https://github.com/<owner>/<repo>/releases/download/v1.0.0/model_metadata.tar.gz

# Reassemble model/
mkdir -p model
for f in 0 1 2 3 4; do tar -xzf fold_$f.tar.gz -C model; done
tar -xzf model_metadata.tar.gz -C model
ls model/   # plans.json, dataset.json, dataset_fingerprint.json, fold_0..4/

# Build + smoke test
docker build -t isles26-mednext:test .
docker run --rm --gpus all -p 4743:4743 \
    -v "$PWD/test/input:/input:ro" \
    -v "$PWD/test/output:/output" \
    isles26-mednext:test &
sleep 30
curl http://localhost:4743/health
# Should return 200 OK
```