import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Compatible with nnU-Net target format: (B, 1, X, Y, Z) float tensor with class indices.
    """
    def __init__(self, alpha=None, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha          # per-class weight tensor, e.g. tensor([0.25, 0.75])
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs:  (B, C, X, Y, Z) logits
        # targets: (B, 1, X, Y, Z) float class indices (nnU-Net convention)
        if targets.ndim == inputs.ndim:
            assert targets.shape[1] == 1
            targets = targets[:, 0]
        targets = targets.long()

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)                     # probability of correct class
        focal_weight = (1.0 - pt) ** self.gamma       # (1-p_t)^γ

        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # mean over non-ignored voxels
        mask = targets != self.ignore_index
        if mask.sum() > 0:
            return focal_loss[mask].mean()
        return focal_loss.mean() * 0.0


class DC_and_Focal_loss(nn.Module):
    """Dice + Focal Loss compound, following nnU-Net's DC_and_CE_loss pattern."""
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_focal=1, weight_dice=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            focal_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        self.focal = FocalLoss(**focal_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label not supported for one-hot targets'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        focal_loss = self.focal(net_output, target) \
            if self.weight_focal != 0 and (self.ignore_label is None or num_fg > 0) else 0

        return self.weight_focal * focal_loss + self.weight_dice * dc_loss


# ── Trainer: Dice + Focal (γ=2.0, standard) ─────────────────────────────
class nnUNetTrainerDiceFocalLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = DC_and_Focal_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'gamma': 2.0},
            weight_focal=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


# ── Trainer: Dice + Focal (γ=3.0, more aggressive) ──────────────────────
class nnUNetTrainerDiceFocalLoss_gamma3(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = DC_and_Focal_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'gamma': 3.0},
            weight_focal=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


# ── Trainer: Pure Focal Loss (no Dice, γ=2.0) ───────────────────────────
class nnUNetTrainerFocalLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = FocalLoss(
            gamma=2.0,
            ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
