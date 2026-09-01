import torch
import numpy as np
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1

# reuse TverskyLoss from the sibling module
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nnUNetTrainerTverskyLoss import TverskyLoss


class DC_Tversky_CE_loss(nn.Module):
    """
    Dice + Tversky + CE — triple compound loss.
    Uses both Dice and Tversky as overlap terms, plus CE.

    Rationale: Dice handles general overlap, Tversky(α=0.3,β=0.7) pushes
    recall, and CE keeps the classification signal.
    """
    def __init__(self, dice_kwargs, tversky_kwargs,
                 weight_dice=1.0, weight_tversky=1.0, weight_ce=1.0,
                 ignore_label=None):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1, **dice_kwargs,
        )
        self.tversky = TverskyLoss(
            apply_nonlin=softmax_helper_dim1, **tversky_kwargs,
        )
        self.ce = RobustCrossEntropyLoss(
            ignore_index=ignore_label if ignore_label is not None else -100,
        )

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_overlap = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_overlap = target
            mask = None

        d_loss = self.dc(net_output, target_overlap, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        t_loss = self.tversky(net_output, target_overlap, loss_mask=mask) \
            if self.weight_tversky != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        return self.weight_ce * ce_loss + self.weight_dice * d_loss + self.weight_tversky * t_loss


# ── Trainer: Dice + Tversky + CE — maximum overlap signal ───────────────
class nnUNetTrainerDiceTverskyCELoss(nnUNetTrainer):
    """Dice + Tversky(α=0.3,β=0.7) + CE — triple compound loss."""
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = DC_Tversky_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp,
             'alpha': 0.3, 'beta': 0.7},
            weight_dice=1.0, weight_tversky=0.5, weight_ce=1.0,
            ignore_label=self.label_manager.ignore_label,
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
