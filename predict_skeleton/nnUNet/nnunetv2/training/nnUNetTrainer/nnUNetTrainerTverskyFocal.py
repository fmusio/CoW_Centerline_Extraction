import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.focal_loss import FocalLoss
from nnunetv2.training.loss.compound_losses import TSKY_and_Focal_loss, FocalTSKY_and_Focal_loss 
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
import numpy as np


class nnUNetTrainerTverskyFocal(nnUNetTrainer):
    # turning mirroring off!
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def _build_loss(self):
        tversky_kwargs = {
            'alpha': 0.75, # punish FN more
            'beta': 0.25,
            'use_log': True,
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }
        focal_kwargs = {
            'alpha': 0.25,
            'gamma': 2,
            'balance_index': 0,
            'smooth': 1e-5,
            'size_average': True
        }
        loss = TSKY_and_Focal_loss(tversky_kwargs, focal_kwargs, weight_focal=2, weight_tversky=1)

        print(f"\nTversky loss with {tversky_kwargs}")
        print(f"\nFocal loss with {focal_kwargs}")
        # print(loss)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # We give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # This gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # We don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # Now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    # def initialize(self, training=True, force_load_plans=False):
    #     super().initialize(training, force_load_plans)
    #     self.loss = self._build_loss()

