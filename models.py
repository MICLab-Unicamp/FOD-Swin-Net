from monai.networks.nets import SwinUNETR

import torch
import torch.nn as nn
from monai.networks.nets import ViT
import torch.nn.functional
from monai.networks.nets import UNet
import torch.optim
import numpy as np
from types import SimpleNamespace

fodlr_final_mean = np.load("utils/fodlr_final_mean.npy")
fodlr_final_std = np.load("utils/fodlr_final_std.npy")
fodgt_final_mean = np.load("utils/fodgt_final_mean.npy")
fodgt_final_std = np.load("utils/fodgt_final_std.npy")


class SwinEncDec(nn.Module):
    def __init__(self, **configs):
        super().__init__()
        self.swin_unet_r_cuda = SwinUNETR(**configs)

    def forward(self, x):
        return self.swin_unet_r_cuda.forward(x)
