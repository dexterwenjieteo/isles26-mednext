"""
nnUNet v2 Trainer for MedNeXt architecture.

Integrates MedNeXt (MICCAI 2023) into the nnU-Net v2 training framework.
Supports all MedNeXt model sizes (S, B, M, L) and kernel sizes (3, 5).

Usage:
    nnUNetv2_train 800 3d_fullres FOLD -tr nnUNetTrainerMedNeXt_S_k3
    nnUNetv2_train 800 3d_fullres FOLD -tr nnUNetTrainerMedNeXt_B_k5

The MedNeXt package must be installed:
    pip install -e $SCRATCH/MedNeXt
"""

from typing import Union, Tuple, List
import torch
from torch import nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerMedNeXt(nnUNetTrainer):
    """
    Base class for MedNeXt trainers.
    Subclasses define model_id and kernel_size.
    """
    model_id = 'S'        # One of S, B, M, L
    kernel_size = 3       # 3 or 5

    @classmethod
    def build_network_architecture(cls, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        from nnunet_mednext import create_mednext_v1

        # Build with deep_supervision=True so the auxiliary output heads
        # (out_1 through out_4) exist in the model. Whether the forward()
        # actually *returns* the list of deep supervision outputs is controlled
        # by model.do_ds, which is toggled via set_deep_supervision_enabled()
        # before training (True) and validation (False).
        model = create_mednext_v1(
            num_input_channels=num_input_channels,
            num_classes=num_output_channels,
            model_id=cls.model_id,
            kernel_size=cls.kernel_size,
            deep_supervision=True,
        )
        model.do_ds = enable_deep_supervision
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        MedNeXt stores deep supervision flag in self.do_ds.
        Toggle it so that training returns a list of outputs (for
        DeepSupervisionWrapper) and validation returns a single tensor.
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.do_ds = enabled


# --- Concrete trainer classes for each MedNeXt variant ---

class nnUNetTrainerMedNeXt_S_k3(nnUNetTrainerMedNeXt):
    model_id = 'S'
    kernel_size = 3


class nnUNetTrainerMedNeXt_S_k5(nnUNetTrainerMedNeXt):
    model_id = 'S'
    kernel_size = 5


class nnUNetTrainerMedNeXt_B_k3(nnUNetTrainerMedNeXt):
    model_id = 'B'
    kernel_size = 3


class nnUNetTrainerMedNeXt_B_k5(nnUNetTrainerMedNeXt):
    model_id = 'B'
    kernel_size = 5


class nnUNetTrainerMedNeXt_M_k3(nnUNetTrainerMedNeXt):
    model_id = 'M'
    kernel_size = 3


class nnUNetTrainerMedNeXt_M_k5(nnUNetTrainerMedNeXt):
    model_id = 'M'
    kernel_size = 5


class nnUNetTrainerMedNeXt_L_k3(nnUNetTrainerMedNeXt):
    model_id = 'L'
    kernel_size = 3


class nnUNetTrainerMedNeXt_L_k5(nnUNetTrainerMedNeXt):
    model_id = 'L'
    kernel_size = 5


# --- 200-epoch sandbox variants -------------------------------------------------

class nnUNetTrainerMedNeXt_S_k3_200epochs(nnUNetTrainerMedNeXt_S_k3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200


# --- 500-epoch sandbox variant --------------------------------------------------

class nnUNetTrainerMedNeXt_S_k3_500epochs(nnUNetTrainerMedNeXt_S_k3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainerMedNeXt_S_k5_200epochs(nnUNetTrainerMedNeXt_S_k5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200
