import torch
import torch.nn as nn
import torch.distributed as dist
import json
import os
from typing import List, Dict, Optional, Tuple, Union

# =====================================================================
# METADATA LOOKUP & UTILITIES
# =====================================================================

_metadata_cache: Optional[dict] = None


def _load_metadata(metadata_path: Optional[str] = None) -> dict:
    """Load the ISLES metadata JSON once and cache it."""
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache
    if metadata_path is None:
        metadata_path = os.path.join(
            os.environ.get("nnUNet_raw", "nnUNet_raw"),
            "Dataset800_ISLES2026",
            "isles_metadata.json",
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. "
            "Run build_isles_metadata.py first."
        )
    with open(metadata_path) as f:
        _metadata_cache = json.load(f)
    return _metadata_cache


def get_num_centers() -> int:
    """Get the number of known centers from metadata."""
    meta = _load_metadata()
    return meta["stats"]["num_centers"] + 1  # +1 for UNKNOWN


def get_metadata_stats() -> dict:
    """Get normalization statistics for continuous features."""
    meta = _load_metadata()
    return meta["stats"]


# =====================================================================
# 1. METADATA ENCODER MODULE
# =====================================================================

class MetadataEncoder(nn.Module):
    """
    Encodes discrete center IDs into dense embeddings and transforms
    continuous clinical features via a learned per-feature linear projection.

    Key design decisions:
    - Uses a Linear (not LayerNorm) so DAYS_POST_STROKE and CHRONICITY
      get independent learned scale/shift, avoiding cross-feature entanglement.
    - Pre-scales raw inputs by dataset-aware upper bounds so both features
      sit roughly in [0, 1] before the linear layer, stabilizing gradients.
    - Includes center dropout for robustness to unseen sites.
    """

    def __init__(
        self,
        num_centers: int = 65,
        embedding_dim: int = 8,
        continuous_dim: int = 2,
        center_dropout_prob: float = 0.1,
        scale: Optional[List[float]] = None,
    ):
        super().__init__()
        # Index num_centers-1 is reserved for 'UNKNOWN' center
        self.num_known_centers = num_centers - 1
        self.center_embedding = nn.Embedding(
            num_embeddings=num_centers, embedding_dim=embedding_dim,
            padding_idx=None,
        )

        # Per-feature linear projection (learned, independent per channel)
        # Initialized near-identity: weight ~ I, bias ~ 0
        self.continuous_proj = nn.Linear(continuous_dim, continuous_dim, bias=True)
        nn.init.eye_(self.continuous_proj.weight)
        nn.init.zeros_(self.continuous_proj.bias)

        # Center dropout: randomly replace center_id with UNKNOWN during training
        self.center_dropout_prob = center_dropout_prob

        # Scaling buffer: divide raw inputs element-wise so each feature ~ [0, 1]
        # Default safe upper bounds: 180 days post stroke, 2 for chronicity (0/1/2)
        default_scale = [180.0, 2.0]
        if scale is not None:
            default_scale = scale
        self.register_buffer("_scale", torch.tensor(default_scale, dtype=torch.float32))

        # Total output dimension
        self.output_dim = embedding_dim + continuous_dim

    def forward(
        self,
        center_ids: torch.Tensor,
        continuous_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            center_ids: LongTensor (B,) — center indices
            continuous_feats: FloatTensor (B, 2) — [DAYS_POST_STROKE, CHRONICITY]
        Returns:
            context_vector: FloatTensor (B, output_dim)
        """
        # --- Center dropout (training only) ---
        if self.training and self.center_dropout_prob > 0.0:
            drop_mask = torch.rand_like(center_ids.float()) < self.center_dropout_prob
            center_ids = torch.where(
                drop_mask,
                torch.full_like(center_ids, self.num_known_centers),  # UNKNOWN index
                center_ids,
            )

        # --- Center embedding: (B,) -> (B, embedding_dim) ---
        center_embeds = self.center_embedding(center_ids)

        # --- Continuous features: scale then project ---
        # 1. Pre-scale raw values to ~[0, 1] using dataset-aware upper bounds
        #    so DAYS_POST_STROKE (1-180+) doesn't overpower CHRONICITY (0/1/2).
        # 2. Learned linear projection gives each feature its own scale & shift.
        #    Explicitly match device & dtype to avoid autocast / stream mismatches.
        scale = self._scale.to(device=continuous_feats.device, dtype=continuous_feats.dtype)
        scaled = continuous_feats / scale.view(1, -1)
        norm_continuous = self.continuous_proj(scaled)

        # --- Concatenate ---
        context_vector = torch.cat([center_embeds, norm_continuous], dim=1)
        return context_vector


# =====================================================================
# 2. FiLM BLOCK MODULE
# =====================================================================

class FiLMBlock(nn.Module):
    """
    Applies Feature-wise Linear Modulation to a 3D feature map
    conditioned on the metadata context vector.
    
    Improvements:
    - Learnable epsilon for numerical stability
    """

    def __init__(
        self,
        num_features: int,
        metadata_dim: int = 10,
        hidden_dim: int = 16,
    ):
        super().__init__()
        self.num_features = num_features

        self.mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features * 2),
        )
        self._init_weights()

    def _init_weights(self):
        """Zero-init final layer so gamma~0 (scale~1.0) and beta~0 initially."""
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, x: torch.Tensor, context_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, D)
            context_vector: (B, metadata_dim)
        Returns:
            modulated_x: (B, C, H, W, D)
        """
        film_params = self.mlp(context_vector)  # (B, C*2)
        gamma, beta = torch.chunk(film_params, chunks=2, dim=1)

        # Reshape to broadcast over spatial dims
        gamma = gamma.view(-1, self.num_features, 1, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1, 1)

        # Affine modulation centered at identity
        return (gamma + 1.0) * x + beta


# =====================================================================
# 3. FiLM RESIDUAL ENCODER U-NET WRAPPER
# =====================================================================

class FiLMResEncUNet(nn.Module):
    """
    Wraps a nnU-Net Residual Encoder U-Net with FiLM conditioning
    on BOTH encoder and decoder stages.
    
    Improvements over original (decoder-only):
    - Adds FiLM to encoder stages (early feature modulation)
    - Adds FiLM at the bottleneck
    - Uses robust skip-connection pairing via zip(reversed(...))
    - Logs FiLM parameter stats for monitoring
    """

    def __init__(
        self,
        base_nnunet_model: nn.Module,
        num_centers: int = 65,
        metadata_dim: int = 10,
        film_on_encoder: bool = True,
        film_on_bottleneck: bool = True,
    ):
        super().__init__()
        self.encoder = base_nnunet_model.encoder
        self.decoder = base_nnunet_model.decoder

        self.metadata_encoder = MetadataEncoder(num_centers=num_centers)

        # --- Build FiLM blocks for encoder stages ---
        self.film_on_encoder = film_on_encoder
        if film_on_encoder:
            self.encoder_film_blocks = nn.ModuleList()
            for stage in self.encoder.stages:
                # nnU-Net ResidualEncoder stages expose .output_channels
                out_ch = stage.output_channels
                self.encoder_film_blocks.append(
                    FiLMBlock(out_ch, metadata_dim=metadata_dim)
                )

        # --- Bottleneck FiLM ---
        self.film_on_bottleneck = film_on_bottleneck
        if film_on_bottleneck:
            # Bottleneck = last skip connection (deepest encoder output, before decoder)
            bottleneck_channels = self.encoder.stages[-1].output_channels
            self.bottleneck_film = FiLMBlock(
                bottleneck_channels, metadata_dim=metadata_dim
            )

        # --- FiLM blocks for decoder stages ---
        self.decoder_film_blocks = nn.ModuleList()
        for stage in self.decoder.stages:
            out_ch = stage.output_channels
            self.decoder_film_blocks.append(
                FiLMBlock(out_ch, metadata_dim=metadata_dim)
            )

    def forward(
        self,
        x: torch.Tensor,
        center_ids: Optional[torch.Tensor] = None,
        continuous_feats: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            x: (B, C_in, H, W, D)
            center_ids: (B,) long, or None to use stored attributes
            continuous_feats: (B, 2) float, or None to use stored attributes
        Returns:
            List[Tensor] — segmentation logits
        """
        # Use stored attributes when called from sliding window (no metadata args)
        if center_ids is None:
            center_ids = getattr(self, "_film_center_ids", None)
        if continuous_feats is None:
            continuous_feats = getattr(self, "_film_continuous_meta", None)

        # Fallback: dummy metadata (UNKNOWN center, zero continuous)
        if center_ids is None:
            center_ids = torch.full(
                (x.shape[0],),
                self.metadata_encoder.num_known_centers,
                dtype=torch.long, device=x.device,
            )
        if continuous_feats is None:
            continuous_feats = torch.zeros(
                (x.shape[0], 2), dtype=torch.float32, device=x.device,
            )

        # 1. Encode metadata
        context = self.metadata_encoder(center_ids, continuous_feats)

        # 2. Encoder forward pass (handles stem + stages internally)
        encoder_outputs = self.encoder(x)  # list of skip connections

        # Apply FiLM to encoder stage outputs if enabled
        if self.film_on_encoder:
            for i in range(len(encoder_outputs)):
                encoder_outputs[i] = self.encoder_film_blocks[i](
                    encoder_outputs[i], context
                )

        # 3. Bottleneck FiLM
        if self.film_on_bottleneck:
            encoder_outputs[-1] = self.bottleneck_film(
                encoder_outputs[-1], context
            )

        # 4. Decoder forward with FiLM injection (mirrors UNetDecoder.forward)
        #    The decoder expects skips: shallowest[0] ... bottleneck[-1]
        seg_outputs = []
        dec_out = encoder_outputs[-1]  # bottleneck
        n_stages = len(self.decoder.stages)

        for s in range(n_stages):
            # Transpose conv upsample
            dec_out = self.decoder.transpconvs[s](dec_out)
            # Concat with corresponding skip (counting from bottleneck backward)
            dec_out = torch.cat((dec_out, encoder_outputs[-(s + 2)]), dim=1)
            # Main conv stage
            dec_out = self.decoder.stages[s](dec_out)
            # FiLM injection after stage processing
            dec_out = self.decoder_film_blocks[s](dec_out, context)
            # Deep supervision heads
            if self.decoder.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[s](dec_out))

        # Match standard nnU-Net output format: high-res first
        if self.decoder.deep_supervision:
            return seg_outputs[::-1]
        else:
            return [self.decoder.seg_layers[-1](dec_out)]

    @torch.no_grad()
    def get_film_stats(self) -> Dict[str, float]:
        """Collect FiLM gamma/beta statistics for monitoring."""
        stats = {}
        for name, block_list in [
            ("enc", getattr(self, "encoder_film_blocks", nn.ModuleList())),
            ("dec", self.decoder_film_blocks),
        ]:
            for i, block in enumerate(block_list):
                # Get the gamma/beta from a sample forward
                # (We use the MLP weights to estimate: mean of last layer)
                w = block.mlp[-1].weight  # (C*2, hidden_dim)
                gamma_w = w[:block.num_features]
                beta_w = w[block.num_features:]
                stats[f"film_{name}{i}_gamma_mean"] = gamma_w.mean().item()
                stats[f"film_{name}{i}_gamma_std"] = gamma_w.std().item()
                stats[f"film_{name}{i}_beta_mean"] = beta_w.mean().item()
                stats[f"film_{name}{i}_beta_std"] = beta_w.std().item()
        return stats


# =====================================================================
# 4. CUSTOM DATALOADER WITH METADATA INJECTION
# =====================================================================

class MetadataDataLoader:
    """
    Wraps a standard nnUNetDataLoader to inject metadata (center_ids,
    continuous_meta) into each batch dict.

    Usage:
        dl_tr = MetadataDataLoader(
            base_dataloader,
            metadata_lookup,
            center_key="center_ids",
            continuous_keys=["days_post_stroke", "chronicity"],
        )
        for batch in dl_tr:
            batch["center_ids"]  # now available
            batch["continuous_meta"]  # now available
    """

    def __init__(
        self,
        base_dataloader,
        metadata_map: Dict[str, dict],
        center_key: str = "center_ids",
        continuous_keys: Tuple[str, str] = ("days_post_stroke", "chronicity"),
    ):
        self.base_dataloader = base_dataloader
        self._data = base_dataloader._data
        self.batch_size = base_dataloader.batch_size
        self._batch_size = getattr(base_dataloader, "_batch_size", None)
        self.number_of_threads_in_multithreaded = getattr(
            base_dataloader, "number_of_threads_in_multithreaded", 1
        )
        self.metadata_map = metadata_map
        self.center_key = center_key
        self.continuous_keys = continuous_keys

    def set_thread_id(self, thread_id):
        """Forwarded to the inner dataloader for batchgenerators compatibility."""
        self.base_dataloader.set_thread_id(thread_id)

    def reset(self):
        """Forwarded to the inner dataloader."""
        self.base_dataloader.reset()

    def get_indices(self):
        return self.base_dataloader.get_indices()

    def __len__(self):
        return len(self.base_dataloader)

    def generate_train_batch(self):
        """
        Generate a batch via the base dataloader, then inject metadata.
        This is called by the batchgenerators infrastructure (including
        background worker processes) instead of __next__.
        """
        batch = self.base_dataloader.generate_train_batch()
        return self._inject_metadata(batch)

    def __next__(self):
        return self.generate_train_batch()

    def _inject_metadata(self, batch: dict) -> dict:
        """Add center_ids and continuous_meta to a batch dict."""
        keys = batch.get("keys", [])
        center_ids = []
        continuous_feats = []

        for k in keys:
            # k is the case identifier, e.g., "ISLES_0000"
            # strip any suffix if present
            case_id = k.split("/")[-1].split(".")[0]
            case_key = case_id if case_id.startswith("ISLES_") else case_id

            if case_key in self.metadata_map:
                meta = self.metadata_map[case_key]
                center_ids.append(meta.get("center_id", -1))
                continuous_feats.append(
                    [meta.get(self.continuous_keys[0], 0.0),
                     meta.get(self.continuous_keys[1], 0.0)]
                )
            else:
                # Fallback: UNKNOWN center, default continuous values
                center_ids.append(-1)
                continuous_feats.append([0.0, 0.0])

        batch["center_ids"] = torch.tensor(center_ids, dtype=torch.long)
        batch["continuous_meta"] = torch.tensor(continuous_feats, dtype=torch.float32)
        return batch


# =====================================================================
# 5. nnU-Net TRAINER WITH FULL FiLM SUPPORT
# =====================================================================

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
    from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
    from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
        NonDetMultiThreadedAugmenter,
    )
    from batchgenerators.dataloading.single_threaded_augmenter import (
        SingleThreadedAugmenter,
    )
    _HAS_NNUNET = True
except ImportError:
    _HAS_NNUNET = False

    class nnUNetTrainer:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class nnUNetTrainerFiLM(nnUNetTrainer):
    """
    Custom nnU-Net Trainer that:
    1. Wraps the base U-Net with FiLMResEncUNet (encoder + decoder FiLM)
    2. Injects clinical metadata (center_id, days_post_stroke, chronicity)
       into each training/validation batch via a custom dataloader wrapper.
    3. Overrides sliding-window inference to pass metadata during validation.
    4. Monitors FiLM parameter statistics via on_epoch_end logging.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        # Load metadata BEFORE super().__init__() so it's available
        # when build_network_architecture is called during initialize().
        if _HAS_NNUNET:
            try:
                full_metadata = _load_metadata()
                self.isles_metadata = full_metadata
                # The JSON has {"metadata": {...}, "stats": {...}} — extract the
                # inner case-level dict for fast lookups.
                self.metadata_map = full_metadata["metadata"]
                self.metadata_stats = full_metadata["stats"]
                print(
                    f"[FiLM] Loaded metadata for {len(self.metadata_map)} cases, "
                    f"{self.metadata_stats['num_centers']} centers."
                )
            except FileNotFoundError:
                print("[FiLM] WARNING: isles_metadata.json not found.")
                print("[FiLM] Training will proceed WITHOUT metadata conditioning.")
                self.isles_metadata = None
                self.metadata_map = {}
                self.metadata_stats = None
        else:
            self.isles_metadata = None
            self.metadata_map = {}
            self.metadata_stats = None

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    # ------------------------------------------------------------------
    # Architecture override
    # ------------------------------------------------------------------

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """
        Build base network from plans, then wrap with FiLMResEncUNet.
        Uses the OLD-style signature matching nnunetv2 2.5.1+computecanada.
        """
        base_network = super().build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return self._wrap_with_film(base_network)

    def _wrap_with_film(self, base_network: nn.Module) -> nn.Module:
        """Wrap a base nnU-Net model with FiLM conditioning modules."""
        # Only call get_num_centers() if metadata was successfully loaded
        if self.metadata_map and self.metadata_stats:
            num_centers = get_num_centers()
        else:
            num_centers = 65

        # Compute dataset-aware scaling factors so both continuous features
        # sit roughly in ~[0, 1] before the learned projection.
        if self.metadata_stats:
            days_scale = (
                self.metadata_stats["days_post_stroke_mean"]
                + 3 * self.metadata_stats["days_post_stroke_std"]
            )
            chron_scale = (
                self.metadata_stats["chronicity_mean"]
                + 3 * self.metadata_stats["chronicity_std"]
            )
            scale = [days_scale, chron_scale]
        else:
            scale = [180.0, 2.0]

        wrapped = FiLMResEncUNet(
            base_nnunet_model=base_network,
            num_centers=num_centers,
            metadata_dim=10,
            film_on_encoder=True,
            film_on_bottleneck=True,
        )
        # Update the registered buffer in-place with dataset-aware values
        wrapped.metadata_encoder._scale.copy_(
            torch.tensor(scale, dtype=torch.float32)
        )
        return wrapped

    # ------------------------------------------------------------------
    # Dataloader override — inject metadata into each batch
    # ------------------------------------------------------------------

    def get_dataloaders(self):
        """
        Create training/validation dataloaders with metadata injection.
        Mirrors the parent nnUNetTrainer.get_dataloaders() but wraps the
        inner dataloaders with MetadataDataLoader.
        """
        if not self.metadata_map:
            print("[FiLM] No metadata available — using standard dataloaders.")
            return super().get_dataloaders()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(self.label_manager.foreground_regions
                     if self.label_manager.has_regions else None),
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(self.label_manager.foreground_regions
                     if self.label_manager.has_regions else None),
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(
                dataset_tr, self.batch_size,
                initial_patch_size, self.configuration_manager.patch_size,
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
                initial_patch_size, self.configuration_manager.patch_size,
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

        # Wrap with MetadataDataLoader
        dl_tr = MetadataDataLoader(dl_tr, self.metadata_map)
        dl_val = MetadataDataLoader(dl_val, self.metadata_map)

        # Wrap in augmenter
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None, pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None, pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    # ------------------------------------------------------------------
    # Metadata sanitization
    # ------------------------------------------------------------------

    def _parse_and_sanitize_metadata(
        self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract metadata from batch with robust error handling and
        dynamic center-ID clamping.
        """
        try:
            center_ids = batch["center_ids"].long().to(self.device, non_blocking=True)
            continuous_meta = (
                batch["continuous_meta"].float().to(self.device, non_blocking=True)
            )
        except KeyError as e:
            raise KeyError(
                f"Missing metadata key '{e.args[0]}' in batch. "
                "Ensure get_dataloaders() properly injects metadata. "
                "Keys in batch: {list(batch.keys())}"
            ) from e

        # Dynamic clamping: use the actual embedding table size
        max_known = self.network.metadata_encoder.num_known_centers
        center_ids = center_ids.clamp(max=max_known)
        return center_ids, continuous_meta

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        center_ids, continuous_meta = self._parse_and_sanitize_metadata(batch)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, enabled=True):
            output = self.network(data, center_ids, continuous_meta)
            loss = self.loss(output, target)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        center_ids, continuous_meta = self._parse_and_sanitize_metadata(batch)

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=True):
                output = self.network(data, center_ids, continuous_meta)
                loss = self.loss(output, target)

        # --- Online Dice computation (matching parent nnUNetTrainer) ---
        # Parent's on_validation_epoch_end expects tp_hard, fp_hard, fn_hard
        # Use highest-resolution output (index 0) if DS is enabled
        if self.enable_deep_supervision:
            output_for_metrics = output[0]
            target_for_metrics = target[0] if isinstance(target, list) else target
        else:
            output_for_metrics = output[0] if isinstance(output, list) else output
            target_for_metrics = target[0] if isinstance(target, list) else target

        axes = [0] + list(range(2, output_for_metrics.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_for_metrics) > 0.5).long()
        else:
            output_seg = output_for_metrics.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output_for_metrics.shape, device=output_for_metrics.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_metrics != self.label_manager.ignore_label).float()
                target_for_metrics[target_for_metrics == self.label_manager.ignore_label] = 0
            else:
                if target_for_metrics.dtype == torch.bool:
                    mask = ~target_for_metrics[:, -1:]
                else:
                    mask = 1 - target_for_metrics[:, -1:]
                target_for_metrics = target_for_metrics[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_for_metrics, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]  # skip background
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }

    # ------------------------------------------------------------------
    # Validation override — pass metadata to network for sliding window
    # ------------------------------------------------------------------

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Injects metadata per-case before sliding-window prediction, then
        exports and computes metrics (same as parent).
        Metadata is stored as network attributes so FiLMResEncUNet.forward()
        picks it up automatically during sliding-window calls.
        """
        if not self.metadata_map:
            self.print_to_log_file("[FiLM] No metadata map — standard validation.")
            return super().perform_actual_validation(save_probabilities)

        self.set_deep_supervision_enabled(False)
        self.network.eval()

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from batchgenerators.utilities.file_and_folder_operations import (
            maybe_mkdir_p, join,
        )
        from nnunetv2.utilities.helpers import empty_cache
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
        from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
        from nnunetv2.configuration import default_num_processes

        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True, device=self.device,
        )
        predictor.manual_initialization(
            self.network, self.plans_manager, self.configuration_manager,
            None, self.dataset_json, self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        _, val_keys = self.do_split()
        if self.is_ddp:
            val_keys = val_keys[self.local_rank :: dist.get_world_size()]

        if self.local_rank == 0:
            self.print_to_log_file(
                f"[FiLM] Validating {len(val_keys)} cases with metadata."
            )

        validation_output_folder = join(self.output_folder, "validation")
        maybe_mkdir_p(validation_output_folder)

        dataset_val = nnUNetDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )

        with torch.no_grad():
            for k in dataset_val.keys():
                self.print_to_log_file(f"[FiLM] predicting {k}")
                data, _, properties = dataset_val.load_case(k)
                data = data[:]
                data = torch.from_numpy(data)

                # --- Resolve metadata for this case ---
                case_key = k.split("/")[-1].split(".")[0]
                if case_key in self.metadata_map:
                    meta = self.metadata_map[case_key]
                    center_ids = torch.tensor(
                        [meta.get("center_id", -1)], dtype=torch.long,
                        device=self.device,
                    )
                    continuous_meta = torch.tensor(
                        [[meta.get("days_post_stroke", 0.0),
                          meta.get("chronicity", 0.5)]],
                        dtype=torch.float32, device=self.device,
                    )
                else:
                    center_ids = torch.tensor(
                        [-1], dtype=torch.long, device=self.device,
                    )
                    continuous_meta = torch.tensor(
                        [[0.0, 0.5]], dtype=torch.float32, device=self.device,
                    )

                # Dynamic clamping
                max_known = self.network.metadata_encoder.num_known_centers
                center_ids = center_ids.clamp(max=max_known)

                # Store on network — FiLMResEncUNet.forward() reads these
                self.network._film_center_ids = center_ids
                self.network._film_continuous_meta = continuous_meta

                # Predict + export (same as parent)
                output_filename_truncated = join(validation_output_folder, k)
                prediction = predictor.predict_sliding_window_return_logits(data)
                export_prediction_from_logits(
                    prediction.cpu(), properties,
                    self.configuration_manager, self.plans_manager,
                    self.dataset_json, output_filename_truncated,
                    save_probabilities,
                )
                empty_cache(self.device)

        # Clean up
        for attr in ["_film_center_ids", "_film_continuous_meta"]:
            if hasattr(self.network, attr):
                delattr(self.network, attr)

        # Metrics (same as parent)
        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
                validation_output_folder,
                join(validation_output_folder, "summary.json"),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                (self.label_manager.foreground_regions
                 if self.label_manager.has_regions
                 else self.label_manager.foreground_labels),
                self.label_manager.ignore_label,
                chill=True,
                num_processes=default_num_processes * dist.get_world_size()
                if self.is_ddp else default_num_processes,
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file(
                "Mean Validation Dice: ",
                metrics["foreground_mean"]["Dice"],
                also_print_to_console=True,
            )

        self.set_deep_supervision_enabled(True)

    # ------------------------------------------------------------------
    # Epoch-end logging — monitor FiLM parameters
    # ------------------------------------------------------------------

    def on_epoch_end(self):
        """Log FiLM parameter statistics at the end of each epoch."""
        super().on_epoch_end()

        if hasattr(self, "network") and hasattr(self.network, "get_film_stats"):
            try:
                stats = self.network.get_film_stats()
                writer = getattr(self, "writer", None)
                if self.local_rank == 0 and writer is not None:
                    for name, val in stats.items():
                        writer.add_scalar(
                            f"film/{name}", val, self.current_epoch
                        )
            except Exception as e:
                if self.local_rank == 0:
                    print(f"[FiLM] Could not log film stats: {e}")

    # ------------------------------------------------------------------
    # Configuration overrides
    # ------------------------------------------------------------------

    @property
    def tile_step_size(self) -> float:
        """Sliding window step size as fraction of patch size."""
        return 0.5