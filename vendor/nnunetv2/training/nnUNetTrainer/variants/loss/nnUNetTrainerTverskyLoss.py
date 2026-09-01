import torch
import numpy as np
from torch import nn

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class TverskyLoss(nn.Module):
    """
    Tversky Loss — asymmetric generalisation of Dice.

    Tversky Index = TP / (TP + α*FP + β*FN)

    - α=β=0.5  → standard Dice
    - α=0.3, β=0.7 → penalise false negatives more (lesion detection focus)
    - α=0.7, β=0.3 → penalise false positives more (precision focus)

    Mimics MemoryEfficientSoftDiceLoss API so it can drop into compound losses.
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.0,
                 ddp=True, alpha=0.3, beta=0.7):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            tp = (x * y_onehot).sum(axes)
            fp = (x * (~y_onehot)).sum(axes)
            fn = ((1.0 - x) * y_onehot).sum(axes)
        else:
            tp = (x * y_onehot * loss_mask).sum(axes)
            fp = (x * (~y_onehot) * loss_mask).sum(axes)
            fn = ((1.0 - x) * y_onehot * loss_mask).sum(axes)

        # Tversky: TI = (tp + s) / (tp + α*fp + β*fn + s)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        ti = ti.mean()
        return -ti


class Tversky_and_CE_loss(nn.Module):
    """
    Tversky + Cross-Entropy compound loss.
    Tversky replaces Dice as the overlap term.
    """
    def __init__(self, tversky_kwargs, weight_tversky=1, weight_ce=1,
                 ignore_label=None):
        super().__init__()
        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(
            ignore_index=ignore_label if ignore_label is not None else -100,
        )
        self.tversky = TverskyLoss(apply_nonlin=softmax_helper_dim1, **tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label not supported for one-hot targets'
            mask = target != self.ignore_label
            target_tv = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_tv = target
            mask = None

        tv_loss = self.tversky(net_output, target_tv, loss_mask=mask) \
            if self.weight_tversky != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        return self.weight_ce * ce_loss + self.weight_tversky * tv_loss


# ── Trainer: Tversky + CE  (α=0.3, β=0.7 — penalise FN, lesion detection) ──
class nnUNetTrainerTverskyCELoss(nnUNetTrainer):
    """Tversky(α=0.3,β=0.7) + CE — favours recall over precision."""
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = Tversky_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp,
             'alpha': 0.3, 'beta': 0.7},
            weight_tversky=1, weight_ce=1,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


# ── Trainer: Tversky + CE  (α=0.7, β=0.3 — penalise FP, precision focus) ──
class nnUNetTrainerTverskyCELoss_precision(nnUNetTrainer):
    """Tversky(α=0.7,β=0.3) + CE — favours precision over recall."""
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = Tversky_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp,
             'alpha': 0.7, 'beta': 0.3},
            weight_tversky=1, weight_ce=1,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


# ── Trainer: Pure Tversky  (α=0.3, β=0.7) — no CE ──────────────────────
class nnUNetTrainerTverskyLoss(nnUNetTrainer):
    """Pure Tversky(α=0.3,β=0.7) — overlap-only, recall-focused."""
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = TverskyLoss(
            apply_nonlin=softmax_helper_dim1,
            batch_dice=self.configuration_manager.batch_dice,
            do_bg=False, smooth=1e-5, ddp=self.is_ddp,
            alpha=0.3, beta=0.7,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
