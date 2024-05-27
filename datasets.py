import os
import numpy as np
from utils.load_mem_map import MemMap
from utils.read_dataset import ReadDataset
from torch.utils.data import Dataset


class ReadDataCropPre(Dataset):
    def __init__(self, **kwargs):
        self.size_3d_patch = 9
        self.len_dataset = 200
        self.element = []
        self.step_per_subject = 0
        self.data_list_id_ = None
        self.read_dataset = ReadDataset(kwargs["size_3d_patch"])
        self.count_subjects = None

    def __get_mask(self, path, name):
        coordinate = MemMap.read_mem_map(f"{path}/coordinates/{name}", data_dtype=np.dtype("int64"),
                                         data_shape=(6,))

        return coordinate

    def __getitem__(self, index):
        dir_fod_sample = self.data_list_id_[self.step_per_subject % self.count_subjects]

        fod_sample = self.read_dataset.get_sample_subject_memmap_no_patch(dir_fod_sample)

        path_coordinates = "../coordinates_train"

        id_select = dir_fod_sample.replace(
            "../train/", "")

        ids = os.listdir(f"{path_coordinates}/{id_select}/coordinates/")

        idx_rand = np.random.randint(0, len(ids))

        coordinate = self.__get_mask(f"{path_coordinates}/{id_select}", ids[idx_rand])

        x_start, x_end, y_start, y_end, z_start, z_end = coordinate

        fodgt_3D_patches = fod_sample["fodgt_3D_patches"][x_start: x_end, y_start: y_end, z_start: z_end, :]
        fodlr_3D_patches = fod_sample["fodlr_3D_patches"][x_start: x_end, y_start: y_end, z_start: z_end, :]

        self.step_per_subject += 1

        data_dict = {'fodlr': fodlr_3D_patches.transpose(3, 0, 1, 2).astype(np.float32),
                     'fodgt': fodgt_3D_patches.transpose(3, 0, 1, 2).astype(np.float32)}

        return data_dict

    def __len__(self):
        return self.len_dataset
