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
    nnUNetTrainerDiceTopK1Loss,
    nnUNetTrainerDiceTopK5Loss,
    nnUNetTrainerDiceTopK10Loss,
    nnUNetTrainerTopk1Loss,
    nnUNetTrainerTopk5Loss,
    nnUNetTrainerTopk10Loss,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCELoss import (
    nnUNetTrainerCELoss,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _EpochLimitMixin:
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200


# ── 200-epoch variants ───────────────────────────────────────────

class nnUNetTrainer_200epochs(_EpochLimitMixin, nnUNetTrainer):
    pass

class nnUNetTrainerDiceFocalLoss_200epochs(_EpochLimitMixin, nnUNetTrainerDiceFocalLoss):
    pass

class nnUNetTrainerDiceFocalLoss_gamma3_200epochs(_EpochLimitMixin, nnUNetTrainerDiceFocalLoss_gamma3):
    pass

class nnUNetTrainerFocalLoss_200epochs(_EpochLimitMixin, nnUNetTrainerFocalLoss):
    pass

class nnUNetTrainerTverskyCELoss_200epochs(_EpochLimitMixin, nnUNetTrainerTverskyCELoss):
    pass

class nnUNetTrainerTverskyCELoss_precision_200epochs(_EpochLimitMixin, nnUNetTrainerTverskyCELoss_precision):
    pass

class nnUNetTrainerTverskyLoss_200epochs(_EpochLimitMixin, nnUNetTrainerTverskyLoss):
    pass

class nnUNetTrainerDiceWeightedCELoss_200epochs(_EpochLimitMixin, nnUNetTrainerDiceWeightedCELoss):
    pass

class nnUNetTrainerDiceWeightedCELoss_strong_200epochs(_EpochLimitMixin, nnUNetTrainerDiceWeightedCELoss_strong):
    pass

class nnUNetTrainerDiceWeightedCELoss_extreme_200epochs(_EpochLimitMixin, nnUNetTrainerDiceWeightedCELoss_extreme):
    pass

class nnUNetTrainerDiceTverskyCELoss_200epochs(_EpochLimitMixin, nnUNetTrainerDiceTverskyCELoss):
    pass

class nnUNetTrainerDiceLoss_200epochs(_EpochLimitMixin, nnUNetTrainerDiceLoss):
    pass

class nnUNetTrainerDiceCELoss_noSmooth_200epochs(_EpochLimitMixin, nnUNetTrainerDiceCELoss_noSmooth):
    pass

class nnUNetTrainerDiceTopK1Loss_200epochs(_EpochLimitMixin, nnUNetTrainerDiceTopK1Loss):
    pass

class nnUNetTrainerDiceTopK5Loss_200epochs(_EpochLimitMixin, nnUNetTrainerDiceTopK5Loss):
    pass

class nnUNetTrainerDiceTopK10Loss_200epochs(_EpochLimitMixin, nnUNetTrainerDiceTopK10Loss):
    pass

class nnUNetTrainerTopk1Loss_200epochs(_EpochLimitMixin, nnUNetTrainerTopk1Loss):
    pass

class nnUNetTrainerTopk5Loss_200epochs(_EpochLimitMixin, nnUNetTrainerTopk5Loss):
    pass

class nnUNetTrainerTopk10Loss_200epochs(_EpochLimitMixin, nnUNetTrainerTopk10Loss):
    pass

class nnUNetTrainerCELoss_200epochs(_EpochLimitMixin, nnUNetTrainerCELoss):
    pass
