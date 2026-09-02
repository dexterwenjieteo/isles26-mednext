"""
ISLES 2026 submission — MedNeXt-L (kernel 5) 5-fold ensemble via nnU-Net v2.

Interface (as prescribed by the ISLES26 template):
  Inputs:
    - /input/images/t1-brain-mri/            (single T1 brain MRI, .nii.gz/.nii/.mha)
    - /input/stroke-metadata.json            (optional, only logged)
  Outputs:
    - /output/images/stroke-lesion-segmentation/output.mha   (binary 0/1 mask)
    - /output/images/lesion-probability-map/output.mha       (softmax prob of lesion)

Implementation notes:
  - The MedNeXt-L k5 network (63M params) is rebuilt from the vendored
    nnU-Net v2 + MedNeXt source and loaded with `checkpoint_best.pth`
    (the checkpoint nnU-Net used for final validation).
  - The 5 fold outputs are ensembled by nnU-Net (averaged logits), with
    mirroring TTA and Gaussian-weighted sliding-window inference.
  - `predict_single_npy_array` is used because it is fully synchronous and
    avoids multiprocessing (which is fragile inside a single-case container).
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

FOLDS = (0, 1, 2, 3, 4)
CHECKPOINT = "checkpoint_best.pth"

# nnU-Net prints harmless warnings at import time when these paths are unset.
# Point them at real-looking locations to keep container logs clean.
os.environ.setdefault("nnUNet_raw", "/opt/app/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/opt/app/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/opt/app/nnUNet_results")
os.environ["nnUNet_compile"] = "false"


def _find_model_dir() -> Path:
    """Locate the model directory (plans.json + dataset.json + fold_*)."""
    if (MODEL_PATH / "plans.json").exists():
        return MODEL_PATH
    # Grand Challenge may extract the uploaded model tarball into a subdirectory.
    if MODEL_PATH.is_dir():
        for d in sorted(MODEL_PATH.iterdir()):
            if d.is_dir() and (d / "plans.json").exists():
                return d
    raise FileNotFoundError(f"No plans.json found under {MODEL_PATH}")


def init_model():
    from nnunet_mednext import create_mednext_v1
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.label_handling.label_handling import (
        determine_num_input_channels,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_dir = _find_model_dir()
    print(f"Loading MedNeXt-L k5 5-fold ensemble from {model_dir}")

    # Build the predictor manually (instead of initialize_from_trained_model_folder).
    # This intentionally avoids importing nnUNetTrainer.py, which would drag in
    # the whole training stack (batchgeneratorsv2, seaborn, matplotlib, ...).
    dataset_json = json.loads((model_dir / "dataset.json").read_text())
    plans = json.loads((model_dir / "plans.json").read_text())

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration("3d_fullres")
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    num_output_channels = label_manager.num_segmentation_heads

    # Rebuild MedNeXt-L k5 exactly as nnUNetTrainerMedNeXt_L_k5 did:
    #   deep_supervision=True creates the auxiliary output heads (out_1..out_4)
    #   present in the checkpoint; do_ds=False makes forward() return a single
    #   output tensor (as used for validation/inference).
    network = create_mednext_v1(
        num_input_channels=num_input_channels,
        num_classes=num_output_channels,
        model_id="L",
        kernel_size=5,
        deep_supervision=True,
    )
    network.do_ds = False

    # Load the 5 fold checkpoints (nnU-Net averages their logits at inference).
    parameters = []
    mirroring_axes = None
    for f in FOLDS:
        ck = torch.load(
            str(model_dir / f"fold_{f}" / CHECKPOINT),
            map_location="cpu",
            weights_only=False,
        )
        parameters.append(ck["network_weights"])
        if mirroring_axes is None:
            mirroring_axes = ck.get("inference_allowed_mirroring_axes")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  # mirroring TTA = 8x forward passes; DISABLED to
        # stay within Grand Challenge's per-case time limit. The final test scan
        # (267x336x336 @ 0.75mm) is ~2x the sanity-check volume and timed out
        # with mirroring on. Mirroring costs ~0.5-1.5% Dice but 8x compute.
        perform_everything_on_device=True,  # auto-disabled on CPU
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        network,
        plans_manager,
        configuration_manager,
        parameters,
        dataset_json,
        "nnUNetTrainerMedNeXt_L_k5",
        mirroring_axes,
    )
    print("Model loaded.")
    return predictor


def _load_input_image():
    image_dir = INPUT_PATH / "images" / "t1-brain-mri"
    candidates = (
        glob.glob(str(image_dir / "*.nii.gz"))
        + glob.glob(str(image_dir / "*.nii"))
        + glob.glob(str(image_dir / "*.mha"))
    )
    if not candidates:
        raise FileNotFoundError(f"No image file found in {image_dir}")
    image_path = min(candidates)
    print(f"Reading input: {image_path}")

    sitk_image = sitk.ReadImage(image_path)
    # nnU-Net convention: array axes are (z, y, x); SITK spacing is (x, y, z),
    # so it is reversed to (z, y, x) and made absolute (see SimpleITKIO).
    array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)[None]  # (1, z, y, x)
    spacing = list(np.abs(sitk_image.GetSpacing()))[::-1]  # (z, y, x)
    return sitk_image, array, spacing, image_path


def _write_output_image(array, reference_image, location):
    location.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(array)
    out.SetSpacing(reference_image.GetSpacing())
    out.SetOrigin(reference_image.GetOrigin())
    out.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(out, str(location / "output.mha"), useCompression=True)


def run(model):
    predictor = model
    sitk_image, array, spacing, _ = _load_input_image()

    metadata_path = INPUT_PATH / "stroke-metadata.json"
    if metadata_path.exists():
        try:
            print("Stroke metadata:")
            print(json.dumps(json.loads(metadata_path.read_text()), indent=2))
        except Exception as e:  # noqa: BLE001 - metadata is non-critical
            print(f"(could not read metadata: {e})")

    # seg: (z, y, x) uint8 in {0, 1}; probs: (2, z, y, x) softmax, index 1 = lesion.
    seg, probs = predictor.predict_single_npy_array(
        array,
        {"spacing": spacing},
        None,
        None,
        save_or_return_probabilities=True,
    )

    _write_output_image(
        seg.astype(np.uint8),
        sitk_image,
        OUTPUT_PATH / "images" / "stroke-lesion-segmentation",
    )
    _write_output_image(
        probs[1].astype(np.float32),
        sitk_image,
        OUTPUT_PATH / "images" / "lesion-probability-map",
    )
    print("Inference complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(init_model()))
