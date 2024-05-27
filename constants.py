from utils.util import set_device
from utils.save_models import SaveBestModel
from datasets import ReadDataCropPre
import torch
from models import SwinEncDec


DEVICE = set_device()

save_best_model = SaveBestModel()

FACTORY_DICT = {
    "model": {
        "SwinEncDec": SwinEncDec,
    },
    "dataset": {
        "ReadDataCropPre": ReadDataCropPre,
    },
    "transformation": {
    },
    "optimizer": {
        "Adam": torch.optim.Adam
    },
    "loss": {
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
        "MSELoss": torch.nn.MSELoss(),
        "MAE": torch.nn.L1Loss(),
    },
}


class FactoryRand:
    @staticmethod
    def call_rand_split(name, **kwargs):
        return eval(name)(**kwargs)
