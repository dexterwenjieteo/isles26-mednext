import torch

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceFocalLoss import (
    nnUNetTrainerDiceFocalLoss,
    nnUNetTrainerDiceFocalLoss_gamma3,
    nnUNetTrainerFocalLoss,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTverskyLoss import (
    nnUNetTrainerTverskyCELoss,
    nnUNetTrainerTverskyCELoss_precision,
    nnUNetTrainerTverskyLoss,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceWeightedCELoss import (
    nnUNetTrainerDiceWeightedCELoss,
    nnUNetTrainerDiceWeightedCELoss_strong,
    nnUNetTrainerDiceWeightedCELoss_extreme,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceTverskyCELoss import (
    nnUNetTrainerDiceTverskyCELoss,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import (
    nnUNetTrainerDiceLoss,
    nnUNetTrainerDiceCELoss_noSmooth,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import (
    nnUNetTrainerDiceTopK10Loss,
    nnUNetTrainerTopk10Loss,
    nnUNetTrainerDiceTopK10Loss,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCELoss import (
    nnUNetTrainerCELoss,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _EpochLimitMixin:
    """Set num_epochs via __init__ so it doesn't interfere with static attributes."""
    def __init__(self, *args, num_epochs=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = num_epochs


# ── 100-epoch variants of every loss trainer ──────────────────────────

class nnUNetTrainer_100epochs(_EpochLimitMixin, nnUNetTrainer):
    pass

class nnUNetTrainerDiceFocalLoss_100epochs(_EpochLimitMixin, nnUNetTrainerDiceFocalLoss):
    pass

class nnUNetTrainerDiceFocalLoss_gamma3_100epochs(_EpochLimitMixin, nnUNetTrainerDiceFocalLoss_gamma3):
    pass

class nnUNetTrainerFocalLoss_100epochs(_EpochLimitMixin, nnUNetTrainerFocalLoss):
    pass

class nnUNetTrainerTverskyCELoss_100epochs(_EpochLimitMixin, nnUNetTrainerTverskyCELoss):
    pass

class nnUNetTrainerTverskyCELoss_precision_100epochs(_EpochLimitMixin, nnUNetTrainerTverskyCELoss_precision):
    pass

class nnUNetTrainerTverskyLoss_100epochs(_EpochLimitMixin, nnUNetTrainerTverskyLoss):
    pass

class nnUNetTrainerDiceWeightedCELoss_100epochs(_EpochLimitMixin, nnUNetTrainerDiceWeightedCELoss):
    pass

class nnUNetTrainerDiceWeightedCELoss_strong_100epochs(_EpochLimitMixin, nnUNetTrainerDiceWeightedCELoss_strong):
    pass

class nnUNetTrainerDiceWeightedCELoss_extreme_100epochs(_EpochLimitMixin, nnUNetTrainerDiceWeightedCELoss_extreme):
    pass

class nnUNetTrainerDiceTverskyCELoss_100epochs(_EpochLimitMixin, nnUNetTrainerDiceTverskyCELoss):
    pass

class nnUNetTrainerDiceLoss_100epochs(_EpochLimitMixin, nnUNetTrainerDiceLoss):
    pass

class nnUNetTrainerDiceCELoss_noSmooth_100epochs(_EpochLimitMixin, nnUNetTrainerDiceCELoss_noSmooth):
    pass

class nnUNetTrainerDiceTopK10Loss_100epochs(_EpochLimitMixin, nnUNetTrainerDiceTopK10Loss):
    pass

class nnUNetTrainerTopk10Loss_100epochs(_EpochLimitMixin, nnUNetTrainerTopk10Loss):
    pass

class nnUNetTrainerCELoss_100epochs(_EpochLimitMixin, nnUNetTrainerCELoss):
    pass
