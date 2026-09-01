"""
On-the-fly CarveMix augmentation trainer for nnU-Net v2.

Dynamic version: pastes real lesion patches into healthy brain tissue
of training samples during data loading. This provides infinite diversity
(unlike the offline pre-generated D806) without dataset bloat.

Trainer class: nnUNetTrainerMedNeXt_L_k5_CarveMix

Usage:
    nnUNetv2_train 800 3d_fullres 0 -tr nnUNetTrainerMedNeXt_L_k5_CarveMix
"""

import os
import json
import numpy as np
import torch

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from scipy.ndimage import distance_transform_edt

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMedNeXt import (
    nnUNetTrainerMedNeXt_L_k5,
)
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.utils import get_case_identifiers
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.paths import nnUNet_raw


class CarveMixOnTheFlyTransform(BasicTransform):
    """
    On-the-fly CarveMix: pastes a lesion patch from a pre-loaded donor bank
    into healthy brain tissue of training patches.

    Operates on single-patch data dicts (as called by nnUNetDataLoader3D),
    BEFORE deep supervision downsampling so all DS levels reflect the
    synthesized lesion.

    Keys expected: 'image' (C,D,H,W tensor), 'segmentation' (D,H,W tensor)

    Thread-safety: uses np.random (each multiprocessing worker gets its own
    seed via fork on Linux, so no manual seed reset needed).
    """

    def __init__(
        self,
        lesion_bank: list,
        apply_prob: float = 0.4,
        blend_sigma: float = 3.0,
        donor_weights: np.ndarray = None,
    ):
        super().__init__()
        self.lesion_bank = lesion_bank
        self.apply_prob = apply_prob
        self.blend_sigma = blend_sigma
        self.donor_weights = donor_weights

    def apply(self, data_dict, **params):
        # Skip with (1 - apply_prob) chance, or if bank is empty
        if np.random.random() > self.apply_prob:
            return data_dict
        if not self.lesion_bank:
            return data_dict

        img = data_dict['image'].numpy()         # (C, D, H, W) float32
        seg = data_dict['segmentation'].numpy()  # (1, D, H, W) or (D, H, W)
        # nnUNet may pass seg with a leading singleton channel dim
        if seg.ndim == 4 and seg.shape[0] == 1:
            seg = seg[0]

        C, D, H, W = img.shape

        # Pick a donor from the bank (weighted sampling if donor_weights set)
        if self.donor_weights is not None:
            donor_idx = np.random.choice(
                len(self.lesion_bank), p=self.donor_weights)
        else:
            donor_idx = np.random.randint(0, len(self.lesion_bank))
        donor = self.lesion_bank[donor_idx]
        d_img = donor['img']   # (C, pd, ph, pw)
        d_lbl = donor['lbl']   # (pd, ph, pw)

        # Match channel count (safety: handles multi-channel datasets)
        if d_img.shape[0] != C:
            if d_img.shape[0] > C:
                d_img = d_img[:C]
            else:
                pad = np.zeros((C - d_img.shape[0],) + d_img.shape[1:],
                               dtype=np.float32)
                d_img = np.concatenate([d_img, pad], axis=0)

        pd, ph, pw = d_lbl.shape

        # Find healthy brain region: non-lesion, non-padding tissue.
        # Preprocessed seg: -1 = outside/padding, 0 = healthy+background,
        # 1 = lesion. Paste into seg == 0 (any non-lesion brain).
        brain_mask = seg == 0
        if not brain_mask.any():
            return data_dict  # no healthy tissue in this patch

        # Rejection-sample the paste center: uniform over healthy-brain
        # voxels, but O(tries) instead of a full-volume np.argwhere (which
        # allocated ~2M-element coordinate arrays per 128^3 patch and
        # dominated the epoch wall-time). On brain patches the hit rate is
        # high, so ~2-3 probes on average; capped at 50 to be safe.
        center = None
        for _ in range(50):
            z = np.random.randint(0, D)
            y = np.random.randint(0, H)
            x = np.random.randint(0, W)
            if brain_mask[z, y, x]:
                center = (z, y, x)
                break
        if center is None:
            return data_dict  # could not find healthy tissue in 50 tries

        # Compute paste bounding box (clamp to volume bounds)
        z0 = max(0, center[0] - pd // 2)
        z1 = min(D, z0 + pd)
        z0 = max(0, z1 - pd)
        y0 = max(0, center[1] - ph // 2)
        y1 = min(H, y0 + ph)
        y0 = max(0, y1 - ph)
        x0 = max(0, center[2] - pw // 2)
        x1 = min(W, x0 + pw)
        x0 = max(0, x1 - pw)

        actual_d = z1 - z0
        actual_h = y1 - y0
        actual_w = x1 - x0

        # Crop/pad donor to fit target region
        d_img_crop = np.zeros((C, actual_d, actual_h, actual_w), dtype=np.float32)
        d_lbl_crop = np.zeros((actual_d, actual_h, actual_w), dtype=np.uint8)
        md = min(pd, actual_d)
        mh = min(ph, actual_h)
        mw = min(pw, actual_w)
        d_img_crop[:, :md, :mh, :mw] = d_img[:, :md, :mh, :mw]
        d_lbl_crop[:md, :mh, :mw] = d_lbl[:md, :mh, :mw]

        mask_region = d_lbl_crop > 0
        if mask_region.sum() < 5:
            return data_dict  # too little lesion content in crop

        # Alpha blending with distance transform for smooth edges
        dist = distance_transform_edt(mask_region)
        alpha = np.clip(dist / max(self.blend_sigma, 1.0), 0, 1)
        alpha = alpha[np.newaxis, ...]  # (1, d, h, w) for broadcasting

        # Blend image
        target_region = img[:, z0:z1, y0:y1, x0:x1]
        blended = d_img_crop * alpha + target_region * (1 - alpha)
        img[:, z0:z1, y0:y1, x0:x1] = blended

        # Blend segmentation (union: existing lesion ∪ pasted lesion)
        seg[z0:z1, y0:y1, x0:x1] = np.maximum(
            seg[z0:z1, y0:y1, x0:x1],
            mask_region.astype(seg.dtype),
        )

        # Restore channel dim — downstream transforms (e.g. DownsampleSegForDS)
        # expect segmentation as (1, D, H, W), not (D, H, W)
        if seg.ndim == 3:
            seg = seg[None]

        data_dict['image'] = torch.from_numpy(img)
        data_dict['segmentation'] = torch.from_numpy(seg)
        return data_dict


class nnUNetTrainerMedNeXt_L_k5_CarveMix(nnUNetTrainerMedNeXt_L_k5):
    """
    MedNeXt-L_k5 (1000 epochs) with on-the-fly CarveMix augmentation.

    Builds a lesion bank from preprocessed training data at startup, then
    probabilistically pastes lesion patches into healthy brain regions of
    each training patch during data loading.

    Advantages over offline D806:
      - Infinite diversity (random donor × random location, every patch)
      - No 3× dataset bloat on disk
      - Each original subject still gets ~250 views per epoch

    Expected val Dice improvement: 0.01–0.03 over baseline L_k5 (0.663).
    """

    # When True, donors are sampled by inverse-chronicity-frequency weighting
    # (acute pasted ~33% of the time instead of ~6%). Set False for the
    # uniform-sampling control arm of the acute-prioritization ablation.
    acute_prioritization = True

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self._carvemix_lesion_bank = None
        self._carvemix_donor_weights = None

    # ------------------------------------------------------------------
    #  Chronicity metadata (for acute donor prioritization)
    # ------------------------------------------------------------------

    @staticmethod
    def _bin_chronicity(days):
        """Bin days_post_stroke into 0=acute, 1=subacute, 2=chronic."""
        if days <= 7:
            return 0  # acute (hyperacute)
        elif days <= 180:
            return 1  # subacute
        return 2  # chronic

    def _load_chronicity_map(self):
        """
        Returns {case_id: chronicity_bin} from isles_metadata.json.
        Falls back to None (uniform donor sampling) if metadata is missing.
        """
        try:
            meta_path = os.path.join(
                nnUNet_raw, self.plans_manager.dataset_name,
                'isles_metadata.json')
            with open(meta_path) as f:
                raw = json.load(f)
            metadata = raw['metadata'] if 'metadata' in raw else raw
            return {
                str(case_id): self._bin_chronicity(info['days_post_stroke'])
                for case_id, info in metadata.items()
                if 'days_post_stroke' in info
            }
        except (OSError, KeyError, ValueError) as e:
            self.print_to_log_file(
                f'  WARNING: could not load chronicity metadata '
                f'({e}); falling back to uniform donor sampling.')
            return None

    # ------------------------------------------------------------------
    #  Lesion bank construction
    # ------------------------------------------------------------------

    def _build_lesion_bank(self):
        """
        Pre-load lesion patches from preprocessed training cases.

        Uses memory-mapped .npy files; capped at 500 patches (~500 MB RAM).
        Only extracts from this fold's training split to avoid data leakage.
        """
        if self._carvemix_lesion_bank is not None:
            return

        self._carvemix_lesion_bank = []
        preprocessed_folder = self.preprocessed_dataset_folder
        splits_file = os.path.join(self.preprocessed_dataset_folder_base,
                                   'splits_final.json')

        self.print_to_log_file('Building CarveMix lesion bank...')

        with open(splits_file) as f:
            splits = json.load(f)
        train_cases = [str(c) for c in splits[self.fold]['train']]

        # Get all case identifiers in the preprocessed folder
        all_cases = get_case_identifiers(preprocessed_folder)
        # Filter to training cases only
        train_case_ids = [c for c in all_cases if c in train_cases]

        self.print_to_log_file(
            f'  Scanning {len(train_case_ids)} training cases '
            f'(fold {self.fold})...'
        )

        max_patches = 500  # ~500 MB RAM for lesion bank
        min_lesion_vox = 50
        chronicity_map = self._load_chronicity_map()
        for case_id in train_case_ids:
            if len(self._carvemix_lesion_bank) >= max_patches:
                break

            data_file = os.path.join(preprocessed_folder, f'{case_id}.npy')
            seg_file = os.path.join(preprocessed_folder,
                                    f'{case_id}_seg.npy')

            # Try .npy first (unpacked), fall back to .npz (packed).
            # nnUNet stores npz keys as 'data' and 'seg'.
            if os.path.exists(data_file):
                data = np.load(data_file, mmap_mode='r')
                seg = np.load(seg_file, mmap_mode='r')
            else:
                npz_file = os.path.join(preprocessed_folder, f'{case_id}.npz')
                if not os.path.exists(npz_file):
                    continue
                with np.load(npz_file) as npz:
                    data = npz['data']
                    seg = npz['seg']

            # Squeeze leading dim of 1 for seg — nnUNet stores seg as (1, D, H, W)
            if seg.ndim == 4 and seg.shape[0] == 1:
                seg = seg[0]

            if (seg > 0).sum() < min_lesion_vox:
                continue

            # Extract up to 3 random lesion patches per case
            for _ in range(3):
                if len(self._carvemix_lesion_bank) >= max_patches:
                    break

                lesion_coords = np.argwhere(seg > 0)
                if len(lesion_coords) == 0:
                    break

                center = lesion_coords[np.random.randint(0, len(lesion_coords))]
                half = 32  # 64³ patch

                z0, z1 = (max(0, center[0] - half),
                          min(data.shape[1], center[0] + half))
                y0, y1 = (max(0, center[1] - half),
                          min(data.shape[2], center[1] + half))
                x0, x1 = (max(0, center[2] - half),
                          min(data.shape[3], center[2] + half))

                patch_img = data[:, z0:z1, y0:y1, x0:x1].copy()
                patch_seg = seg[z0:z1, y0:y1, x0:x1].copy()

                if (patch_seg > 0).sum() >= min_lesion_vox:
                    chron = (chronicity_map.get(case_id, 1)
                             if chronicity_map is not None else 1)
                    self._carvemix_lesion_bank.append({
                        'img': patch_img.astype(np.float32),
                        'lbl': (patch_seg > 0).astype(np.uint8),
                        'chron': chron,
                    })

        # ---- acute donor prioritization: inverse-frequency weighting ----
        # Acute is rare (92/1453 ≈ 6%) but is the hardest class. Weight each
        # patch inversely to its chronicity prevalence so acute/subacute/
        # chronic donors are sampled ~equally (1/3 each) at paste time.
        self._carvemix_donor_weights = None
        if (self.acute_prioritization and
                self._carvemix_lesion_bank and chronicity_map is not None):
            import collections
            chron_counts = collections.Counter(
                p['chron'] for p in self._carvemix_lesion_bank)
            n = len(self._carvemix_lesion_bank)
            class_weight = {
                c: (n / cnt if cnt > 0 else 0.0)
                for c, cnt in chron_counts.items()
            }
            raw_weights = np.array(
                [class_weight[p['chron']] for p in self._carvemix_lesion_bank],
                dtype=np.float64)
            self._carvemix_donor_weights = raw_weights / raw_weights.sum()
            self.print_to_log_file(
                f'  CarveMix donor chronicity counts: '
                f'{dict(chron_counts)}')
            self.print_to_log_file(
                f'  Acute-prioritized donor sampling enabled '
                f'(inverse-frequency weights).')

        self.print_to_log_file(
            f'  CarveMix lesion bank: {len(self._carvemix_lesion_bank)} patches'
        )

    # ------------------------------------------------------------------
    #  Dataloader override — inject CarveMix into training transforms
    # ------------------------------------------------------------------

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # ---- training transforms (build standard pipeline first) ----
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        # ---- validation transforms (unchanged) ----
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        # ---- datasets (calls do_split → creates splits_final.json) ----
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # ---- lesion bank (reads splits_final.json — now exists) ----
        self._build_lesion_bank()

        # ---- inject CarveMix right after SpatialTransform, BEFORE the
        # intensity augmentations (noise/brightness/contrast/gamma) ----
        # The donor patch is raw z-scored. Pasting it AFTER intensity augs
        # leaves it intensity-incoherent with the host patch. Pasting into
        # the un-warped image and letting the standard augs act on both
        # donor and host keeps them coherent (same as offline CarveMix).
        if self._carvemix_lesion_bank:
            transforms_list = list(tr_transforms.transforms)

            spat_idx = next(
                (i for i, t in enumerate(transforms_list)
                 if isinstance(t, SpatialTransform)),
                None,
            )
            insert_idx = (spat_idx + 1) if spat_idx is not None else 0

            transforms_list.insert(
                insert_idx,
                CarveMixOnTheFlyTransform(
                    self._carvemix_lesion_bank,
                    apply_prob=0.4,
                    donor_weights=self._carvemix_donor_weights,
                ),
            )

            tr_transforms = ComposeTransforms(transforms_list)

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(
                dataset_tr, self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None,
                transforms=tr_transforms,
            )
            dl_val = nnUNetDataLoader2D(
                dataset_val, self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None,
                transforms=val_transforms,
            )
        else:
            dl_tr = nnUNetDataLoader3D(
                dataset_tr, self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None,
                transforms=tr_transforms,
            )
            dl_val = nnUNetDataLoader3D(
                dataset_val, self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None,
                transforms=val_transforms,
            )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == 'cuda',
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == 'cuda',
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainerMedNeXt_M_k5_CarveMix(nnUNetTrainerMedNeXt_L_k5_CarveMix):
    """
    MedNeXt-M_k5 (1000 epochs) with on-the-fly CarveMix augmentation.

    Same CarveMix logic as the L variant but with the ~3.4x smaller M
    backbone (18.3M params) for faster training under compute constraints.
    """
    model_id = 'M'
    kernel_size = 5


class nnUNetTrainerMedNeXt_M_k5_CarveMix_500epochs(
        nnUNetTrainerMedNeXt_M_k5_CarveMix):
    """
    MedNeXt-M_k5 with on-the-fly CarveMix, 500 epochs.

    Guarantees a fully-completed PolyLR schedule within the compute budget.
    A completed 500-epoch run (LR decays to 0) is preferable to an
    interrupted 1000-epoch run (LR still high at ~72%).
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainerMedNeXt_L_k5_CarveMix_500epochs(
        nnUNetTrainerMedNeXt_L_k5_CarveMix):
    """
    MedNeXt-L_k5 with on-the-fly CarveMix, 500 epochs.

    Full L capacity (63M params) with a fully-converged PolyLR schedule.
    L@1000 (~99h) does not fit the submission deadline; 500 epochs (~50h on
    H100) keeps the paper's claimed backbone while remaining feasible.
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainerMedNeXt_S_k3_CarveMix_500epochs(
        nnUNetTrainerMedNeXt_L_k5_CarveMix):
    """
    MedNeXt-S_k3 with on-the-fly CarveMix (ACUTE-prioritized), 500 epochs.

    Cheap ablation arm: acute donors pasted ~33% of the time.
    """
    model_id = 'S'
    kernel_size = 3

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainerMedNeXt_S_k3_CarveMixUniform_500epochs(
        nnUNetTrainerMedNeXt_S_k3_CarveMix_500epochs):
    """
    MedNeXt-S_k3 with on-the-fly CarveMix (UNIFORM donors), 500 epochs.

    Control arm of the acute-prioritization ablation: donors sampled
    uniformly at random (acute ~6%), otherwise identical.
    """
    acute_prioritization = False


class nnUNetTrainerMedNeXt_L_k5_CarveMixUniform_500epochs(
        nnUNetTrainerMedNeXt_L_k5_CarveMix_500epochs):
    """
    MedNeXt-L_k5 with on-the-fly CarveMix (UNIFORM donors), 500 epochs.

    L-backbone control arm of the acute-prioritization ablation. Donors
    sampled uniformly (acute ~6%); identical to the acute-prioritized L run
    except for donor sampling. Directly comparable to the paper's headline
    backbone.
    """
    acute_prioritization = False
