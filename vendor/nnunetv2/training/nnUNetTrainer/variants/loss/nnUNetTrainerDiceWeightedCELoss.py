import torch
import numpy as np
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with per-class weights to handle class imbalance.

    For ISLES 2026 (2 classes: bg=0, lesion=1), class_weight=[1.0, w_lesion]
    with w_lesion > 1.0 to emphasise the minority class.
    """
    def __init__(self, class_weight=None, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
        else:
            self.class_weight = None

    def forward(self, inputs, targets):
        if targets.ndim == inputs.ndim:
            assert targets.shape[1] == 1
            targets = targets[:, 0]
        targets = targets.long()

        if self.class_weight is not None and self.class_weight.device != inputs.device:
            self.class_weight = self.class_weight.to(inputs.device)

        return nn.functional.cross_entropy(
            inputs, targets,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
        )


class DC_and_WeightedCE_loss(nn.Module):
    """Dice + Weighted CE compound loss."""
    def __init__(self, soft_dice_kwargs, wce_kwargs, weight_ce=1, weight_dice=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            wce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = WeightedCrossEntropyLoss(**wce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


# ── Trainer: Dice + Weighted CE  (class_weight [1, 5] — mild emphasis on lesion) ──
class nnUNetTrainerDiceWeightedCELoss(nnUNetTrainer):
    """Dice + Weighted CE with class_weight=[1, 5] — mild lesion emphasis."""
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = DC_and_WeightedCE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'class_weight': [1.0, 5.0]},
            weight_ce=1, weight_dice=1,
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


# ── Trainer: Dice + Weighted CE  (class_weight [1, 10] — strong emphasis on lesion) ──
class nnUNetTrainerDiceWeightedCELoss_strong(nnUNetTrainer):
    """Dice + Weighted CE with class_weight=[1, 10] — strong lesion emphasis."""
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = DC_and_WeightedCE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'class_weight': [1.0, 10.0]},
            weight_ce=1, weight_dice=1,
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


# ── Trainer: Dice + Weighted CE  (class_weight [1, 20] — extreme emphasis on lesion) ──
class nnUNetTrainerDiceWeightedCELoss_extreme(nnUNetTrainer):
    """Dice + Weighted CE with class_weight=[1, 20] — extreme lesion emphasis."""
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = DC_and_WeightedCE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'class_weight': [1.0, 20.0]},
            weight_ce=1, weight_dice=1,
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
