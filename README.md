# ISLES 2026 — MedNeXt-L (k=5) 5-Fold Ensemble

An inference pipeline for the [ISLES 2026](https://www.isles-challenge.org/) ischemic stroke lesion segmentation challenge, built on a **MedNeXt-L** (kernel size 5) backbone with **nnU-Net v2** training infrastructure and a **5-fold cross-validated ensemble**.

> **Final result:** **4th place** in the ISLES 2026 final test phase, among **46 participating teams**. The 5-fold ensemble achieved a mean validation Dice score of **0.6629** on the official training set.

## Table of contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Architecture](#architecture)
- [Pretrained model weights](#pretrained-model-weights)
- [Running the container](#running-the-container)
- [Inference details](#inference-details)
- [Reproducing training](#reproducing-training)
- [Citation](#citation)
- [License](#license)

## Overview

This repository packages a self-contained Docker container that runs an nnU-Net–style prediction pipeline over the five cross-validation folds and averages their soft predictions.

| Item | Value |
| --- | --- |
| Backbone | MedNeXt-L, kernel size 5 |
| Parameters | ~63 M per fold |
| Folds | 5 (cross-validated) |
| Patch size | 128 × 128 × 128 (full brain) |
| Spacing | 0.75 mm isotropic |
| Tile step size | 0.5 |
| Mean validation Dice | **0.6629** |
| ISLES 2026 final test ranking | **4th of 46 teams** |
| Input modality | T1 brain MRI |
| Output | Per-voxel lesion probability mask (uint8) |

## Repository layout

```
.
├── app.py                       # FastAPI server implementing the Grand Challenge invoke API
├── inference.py                 # Model loading + sliding-window ensemble inference
├── Dockerfile                   # Production container (pytorch 2.13 + cuda 12.6 + cudnn 9)
├── Apptainer.def                # Mirror of the Dockerfile for HPC testing
├── requirements.txt             # Python dependencies
│
├── mednext_arch/                # MedNeXt architecture source (Roy et al., 2023)
├── nnUNetTrainerMedNeXt.py     # nnU-Net v2 trainer (vendored)
├── model/                       # Trained weights, plans, dataset fingerprint (5 folds)
│                                #   ⚠ NOT in git — 482 MB/checkpoint exceeds GitHub's
│                                #   100 MB limit. Download from Releases (see below).
├── vendor/                      # Vendored nnunetv2, dynamic_network_architectures, batchgenerators
│
├── do_build.sh                  # Linux/macOS build helper
├── do_build.ps1                 # Windows PowerShell build helper
├── do_test_run.sh               # Local end-to-end test (HTTP)
├── do_test_apptainer.sh         # Apptainer smoke test
├── do_save.sh                   # Export a Docker image to .tar.gz
├── package_model.sh             # Assemble model/ + vendor/ from $SCRATCH
│
├── test/
│   └── input/                   # Synthetic test input mimicking Grand Challenge layout
│                                #   (no patient data — generated, see below)
│
├── LICENSE                      # MIT
├── CITATION.cff                 # Machine-readable citation metadata
├── README.md                    # This file
└── HANDOFF.md                   # Internal handoff / submission notes
```

## Architecture

The model is a **MedNeXt-L** (large variant, kernel size 5) convolutional network from [Roy et al. (2023)](https://arxiv.org/abs/2303.09975). It uses depth-wise grouped convolutions and a ConvNeXt-style design scaled up to ~63 M parameters for volumetric segmentation. The full-resolution brain is processed in a single forward pass at patch size 128³.

The training scaffold is the standard **nnU-Net v2** trainer (`nnUNetTrainerMedNeXt`), which provides automatic:

- 5-fold cross-validation
- Stochastic gradient descent with warmup
- Poly learning-rate schedule
- Deep supervision
- Mixed-precision (AMP) training
- Data augmentation (rotations, scaling, Gaussian noise, gamma, mirror)

The trainer is vendored into `vendor/` to make the container fully self-contained and to pin exact versions for reproducibility.

## Pretrained model weights

The trained 5-fold ensemble is ~2.4 GB. It is provided as **per-fold GitHub Release assets** rather than regular repo files, because:

- GitHub rejects regular files over **100 MB** (each checkpoint is 482 MB)
- A single combined tarball (2.19 GiB) would exceed the **2 GiB per-file Release limit**

| File | Size | SHA-256 |
| --- | --- | --- |
| `fold_0.tar.gz` | 449 MB | `b01f48cb...` |
| `fold_1.tar.gz` | 449 MB | `b4d1ff9b...` |
| `fold_2.tar.gz` | 448 MB | `321d1f0e...` |
| `fold_3.tar.gz` | 448 MB | `28ab647e...` |
| `fold_4.tar.gz` | 449 MB | `77dd7bcc...` |
| `model_metadata.tar.gz` | 44 KB | `e6965b84...` |

Download all 6 assets from the [Releases page](../../releases), then reassemble:

```bash
mkdir -p model
for f in 0 1 2 3 4; do tar -xzf fold_$f.tar.gz -C model; done
tar -xzf model_metadata.tar.gz -C model
ls model/   # plans.json, dataset.json, dataset_fingerprint.json, fold_0..4/
```

The `model/` directory must sit next to `app.py` and `inference.py` for the container to load it.

> If you retrain the model yourself, the resulting `nnUNet_results/Dataset800_ISLES2026/...` directory **is** this `model/` directory.

## Running the container

### Pull the public image (once published)

```bash
docker pull wenjiede/isles26-mednext:latest
```

### Or build it yourself

```bash
# 1. Make sure model/ is in place (see above)
# 2. Build
docker build -t isles26-mednext:latest .
```

### Test the HTTP endpoint locally

The container exposes a Grand Challenge–compatible FastAPI service on port 4743.

```bash
docker run --rm --gpus all -p 4743:4743 \
    -v "$PWD/test/input:/input:ro" \
    -v "$PWD/test/output:/output" \
    isles26-mednext:latest

# In another terminal:
curl http://localhost:4743/health
```

The bundled `test/input/` contains a **synthetic** 128³ volume (no patient data) that exercises the full pipeline — resampling, sliding-window inference, and output writing. It is small enough to run on CPU in a few minutes.

For the full invoke protocol, see [`HANDOFF.md`](./HANDOFF.md).

## Inference details

The pipeline:

1. **Reads** the T1 brain MRI from `/input/images/t1-brain-mri/`.
2. **Resamples** to 0.75 mm isotropic spacing and crops/zero-pads to the nearest multiple of 128.
3. **Runs sliding-window inference** with a 128³ patch and 0.5 overlap, on each of the 5 folds.
4. **Averages** the fold logits, applies sigmoid, and binarises at the per-fold optimal threshold.
5. **Writes** a uint8 NIfTI mask to `/output/images/ischemic-stroke-lesion-segmentation/`.

Mirroring TTA is **disabled** to stay under the 5-minute-per-job budget on a T4. With mirroring on, the worst-case scan (267 × 336 × 336 voxels) took ~34 minutes. With mirroring off, ~4 minutes — at the cost of approximately 0.5–1.5 % Dice. Five folds are kept because the ensemble matters more than the mirroring for accuracy.

## Reproducing training

Training was performed on a SHARCNET cluster with an NVIDIA T4 (16 GB) on a custom nnU-Net–preprocessed dataset (Dataset 805). Each fold took ~6–8 hours for 500 epochs.

If you have the original training data and want to reproduce:

```bash
# 1. Convert ISLES data to nnU-Net format
nnUNetv2_plan_and_preprocess -d 805 --verify_dataset_integrity -c 3d_fullres

# 2. Train each fold
nnUNetv2_train 805 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMedNeXt
nnUNetv2_train 805 3d_fullres 1 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMedNeXt
# ... etc for folds 2, 3, 4

# 3. Copy the result into model/
cp -r $nnUNet_results/Dataset805_* model/
```

The full training codebase lives in the parent project (separate repository).

## Citation

If you build on this work, please cite the underlying methods:

```bibtex
@article{roy2023mednext,
  title   = {MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation},
  author  = {Roy, Saikat and Koehler, Gregor and Ulrich, Constantin and Baumgartner, Michael and
             Pai, Jingnan and Kleesiek, Jens and Jaeger, Paul and Zhao, Liqiang and
             Maier-Hein, Klaus and Isensee, Fabian and J{\"a}ger, Paul F.},
  journal = {arXiv preprint arXiv:2303.09975},
  year    = {2023}
}

@article{isensee2021nnu-net,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and
             Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021},
  doi     = {10.1038/s41592-020-01008-z}
}
```

A `CITATION.cff` file is also provided for GitHub's "Cite this repository" button.

## License

This project is licensed under the **MIT License** — see [`LICENSE`](./LICENSE) for the full text.

The vendored third-party packages in `vendor/` retain their own respective licenses (nnU-Net v2: Apache 2.0; dynamic-network-architectures: Apache 2.0; batchgenerators: Apache 2.0; MedNeXt: MIT).
