import numpy as np
import torch
import os
import nibabel as nib
import h5py
from utils.load_mem_map import MemMap
# from load_mem_map import MemMap


class ReadDataset:
    def __init__(self, size_3d_path=9):
        self.fod_info = []
        self.margin = int(size_3d_path / 2)
        self.size_3d_patch = size_3d_path

    def just_one_cube(self, fod_info, index_of_patch):

        final_index = fod_info["index_mask"]
        fodlr = fod_info["fodlr"]
        fodgt = fod_info["fodgt"]

        x_start, y_start, z_start = -1, -1, -1
        x_end, y_end, z_end = -1, -1, -1

        while x_start < 0 or y_start < 0 or z_start < 0:

            x = final_index[0, index_of_patch]
            y = final_index[1, index_of_patch]
            z = final_index[2, index_of_patch]

            x_start = x - self.margin
            x_end = x_start + self.size_3d_patch

            y_start = y - self.margin
            y_end = y_start + self.size_3d_patch

            z_start = z - self.margin
            z_end = z_start + self.size_3d_patch

            if x_start < 0 or y_start < 0 or z_start < 0:
                index_of_patch += 1

        fodlr_3D_patches = fodlr[x_start:x_end, y_start:y_end, z_start:z_end, :]
        fodgt_3D_patches = fodgt[x_start:x_end, y_start:y_end, z_start:z_end, :]

        index_of_patch += 1

        return fodlr_3D_patches, fodgt_3D_patches, index_of_patch

    def load_hcp(self, dataset_path, subject_id):

        return self.add_fod_sample(fodlr_path=os.path.join(dataset_path,
                                                           subject_id,
                                                           "HARDI_data",
                                                           "WM_FODs_normalised.nii.gz"),
                                   fodgt_path=os.path.join(dataset_path, subject_id,
                                                           "LARDI_data",
                                                           "normalized_WM_FODs.nii.gz"),
                                   fsl_5ttgen_mask_path=os.path.join(dataset_path,
                                                                     subject_id,
                                                                     "T1_fsl_5ttgen.nii.gz"),
                                   subject_id=subject_id)

    def load_hcp_list(self, dataset_path, subject_id):

        return [self.add_fod_sample(fodlr_path=os.path.join(dataset_path,
                                                            subject_id,
                                                            "HARDI_data",
                                                            "WM_FODs_normalised.nii.gz"),
                                    fodgt_path=os.path.join(dataset_path, subject_id,
                                                            "LARDI_data",
                                                            "normalized_WM_FODs.nii.gz"),
                                    fsl_5ttgen_mask_path=os.path.join(dataset_path,
                                                                      subject_id,
                                                                      "T1_fsl_5ttgen.nii.gz"),
                                    subject_id=subject_id)]

    def add_fod_sample(self, fodlr_path, fodgt_path, fsl_5ttgen_mask_path, subject_id=None):

        try:
            fodlr = nib.load(fodlr_path)
            fodgt = nib.load(fodgt_path)
            fsl5ttgen_mask = nib.load(fsl_5ttgen_mask_path)
        except:
            return None

        fodlr, fixed_fodlr_affine, flipped_axis_fodlr = self.flip_axis_to_match_HCP_space(fodlr.get_fdata(),
                                                                                          fodlr.affine)
        fodgt, fixed_fodgt_affine, flipped_axis_fodgt = self.flip_axis_to_match_HCP_space(fodgt.get_fdata(),
                                                                                          fodgt.affine)

        fsl5ttgen_mask, fixed_fsl5ttgen_mask_affine, flipped_axis_fsl5ttgen_mask = self.flip_axis_to_match_HCP_space(
            fsl5ttgen_mask.get_fdata(),
            fsl5ttgen_mask.affine)
        fsl5ttgen_mask = np.clip(fsl5ttgen_mask, a_min=0.0, a_max=1.0)

        cutted_x, cutted_y, cutted_z, _ = fsl5ttgen_mask.shape

        # Including white matter, cortical grey matter, and subcortical grey matter tissues can improve the performance
        index_mask = np.where(fsl5ttgen_mask[:, :, :, :3].any(axis=-1))
        index_mask = np.asarray(index_mask)
        x = index_mask[0, :]
        y = index_mask[1, :]
        z = index_mask[2, :]
        x_mask = np.logical_and(x >= self.margin, x < (cutted_x - self.margin))
        y_mask = np.logical_and(y >= self.margin, y < (cutted_y - self.margin))
        z_mask = np.logical_and(z >= self.margin, z < (cutted_z - self.margin))
        coord_mask = np.logical_and.reduce([x_mask, y_mask, z_mask])
        self.mask_index = final_index = index_mask[:, coord_mask]  # Apply coord_mask {true or false} in index_mask.
        index_length = len(final_index[0])

        fod_info = {
            'fodlr': fodlr,
            'fodgt': fodgt,
            'subject_id': subject_id,
            'index_mask': final_index,
            'index_length': index_length,
        }
        return fod_info

    @staticmethod
    def flip_axis_to_match_HCP_space(data, affine):

        newAffine = affine.copy()  # could be returned if needed
        flipped_axis = []

        if affine[0, 0] > 0:
            flipped_axis.append("x")
            data = data[::-1, :, :]
            newAffine[0, 0] = newAffine[0, 0] * -1
            newAffine[0, 3] = newAffine[0, 3] * -1

        if affine[1, 1] < 0:
            flipped_axis.append("y")
            data = data[:, ::-1, :]
            newAffine[1, 1] = newAffine[1, 1] * -1
            newAffine[1, 3] = newAffine[1, 3] * -1

        if affine[2, 2] < 0:
            flipped_axis.append("z")
            data = data[:, :, ::-1]
            newAffine[2, 2] = newAffine[2, 2] * -1
            newAffine[2, 3] = newAffine[2, 3] * -1

        return data, newAffine, flipped_axis

    def get_sample_subject(self, subject_path, subject_id=None):

        fodlr, fodgt, fsl5ttgen_mask, lardi_wm_affine, hardi_wm_affine, ttgen_affine = self.read_data(subject_path)

        fodlr, fixed_fodlr_affine, flipped_axis_fodlr = self.flip_axis_to_match_HCP_space(fodlr,
                                                                                          lardi_wm_affine)
        fodgt, fixed_fodgt_affine, flipped_axis_fodgt = self.flip_axis_to_match_HCP_space(fodgt,
                                                                                          hardi_wm_affine)

        fsl5ttgen_mask, fixed_fsl5ttgen_mask_affine, flipped_axis_fsl5ttgen_mask = self.flip_axis_to_match_HCP_space(
            fsl5ttgen_mask,
            ttgen_affine)

        fsl5ttgen_mask = np.clip(fsl5ttgen_mask, a_min=0.0, a_max=1.0)

        cutted_x, cutted_y, cutted_z, _ = fsl5ttgen_mask.shape

        # Including white matter, cortical grey matter, and subcortical grey matter tissues can improve the performance
        index_mask = np.where(fsl5ttgen_mask[:, :, :, :3].any(axis=-1))
        index_mask = np.asarray(index_mask)
        x = index_mask[0, :]
        y = index_mask[1, :]
        z = index_mask[2, :]
        x_mask = np.logical_and(x >= self.margin, x < (cutted_x - self.margin))
        y_mask = np.logical_and(y >= self.margin, y < (cutted_y - self.margin))
        z_mask = np.logical_and(z >= self.margin, z < (cutted_z - self.margin))
        coord_mask = np.logical_and.reduce([x_mask, y_mask, z_mask])
        self.mask_index = final_index = index_mask[:, coord_mask]  # Apply coord_mask {true or false} in index_mask.
        index_length = len(final_index[0])

        fod_info = {
            'fodlr': fodlr,
            'fodgt': fodgt,
            'subject_id': subject_id,
            'index_mask': final_index,
            'index_length': index_length,
        }
        return fod_info

    def get_sample_subject_fastly(self, subject_path, coordinate):

        (fodlr_3D_patches, fodgt_3D_patches,
         fsl5ttgen_mask, lardi_wm_affine,
         hardi_wm_affine, ttgen_affine) = self.read_data_patch(subject_path,
                                                               coordinate)

        fod_info = {
            'fodlr_3D_patches': fodlr_3D_patches,
            'fodgt_3D_patches': fodgt_3D_patches,
        }
        return fod_info

    def read_data_patch(self, path_data, coordinate):

        (x_start, x_end,
         y_start, y_end,
         z_start, z_end) = coordinate

        with h5py.File(f"{path_data}/data.hdf5", "r") as h5_file:
            lardi_wm = h5_file["lardi_wm"]
            hardi_wm = h5_file["hardi_wm"]
            ttgen = h5_file["5ttgen"]

            lardi_wm_affine = h5_file["lardi_wm_affine"]
            hardi_wm_affine = h5_file["hardi_wm_affine"]
            ttgen_affine = h5_file["5ttgen_affine"]

            fodlr_3D_patches = lardi_wm[x_start:x_end, y_start:y_end, z_start:z_end, :]
            fodgt_3D_patches = hardi_wm[x_start:x_end, y_start:y_end, z_start:z_end, :]

            # lardi_wm_data = lardi_wm[:]
            # hardi_wm_data = hardi_wm[:]
            ttgen_data = ttgen[:]
            lardi_wm_affine_data = lardi_wm_affine[:]
            hardi_wm_affine_data = hardi_wm_affine[:]
            ttgen_affine_data = ttgen_affine[:]

        return (fodlr_3D_patches, fodgt_3D_patches, ttgen_data,
                lardi_wm_affine_data, hardi_wm_affine_data, ttgen_affine_data)

    def read_data(self, path_data):
        # (x_start, x_end,
        #  y_start, y_end,
        #  z_start, z_end) = coordinate

        # with h5py.File(f"{path_data}/data.hdf5", "r") as h5_file:
        with h5py.File(f"{path_data}", "r") as h5_file:
            lardi_wm = h5_file["lardi_wm"]
            hardi_wm = h5_file["hardi_wm"]
            ttgen = h5_file["5ttgen"]

            lardi_wm_affine = h5_file["lardi_wm_affine"]
            hardi_wm_affine = h5_file["hardi_wm_affine"]
            ttgen_affine = h5_file["5ttgen_affine"]

            # fodlr_3D_patches = lardi_wm[x_start:x_end, y_start:y_end, z_start:z_end, :]
            # fodgt_3D_patches = hardi_wm[x_start:x_end, y_start:y_end, z_start:z_end, :]

            lardi_wm_data = lardi_wm[:]
            hardi_wm_data = hardi_wm[:]
            ttgen_data = ttgen[:]
            lardi_wm_affine_data = lardi_wm_affine[:]
            hardi_wm_affine_data = hardi_wm_affine[:]
            ttgen_affine_data = ttgen_affine[:]

        return (lardi_wm, hardi_wm, ttgen_data,
                lardi_wm_affine_data, hardi_wm_affine_data, ttgen_affine_data)

    def get_sample_subject_memmap(self, dir_fod_sample, coordinate):

        (x_start, x_end,
         y_start, y_end,
         z_start, z_end) = coordinate

        lardi_wm = MemMap.read_mem_map(f"{dir_fod_sample}/data_lardi.dat")
        hardi_wm = MemMap.read_mem_map(f"{dir_fod_sample}/data_hardi.dat")

        fodlr_3D_patches = lardi_wm[x_start:x_end, y_start:y_end, z_start:z_end, :]
        fodgt_3D_patches = hardi_wm[x_start:x_end, y_start:y_end, z_start:z_end, :]

        fod_info = {
            'fodlr_3D_patches': fodlr_3D_patches,
            'fodgt_3D_patches': fodgt_3D_patches,
        }
        return fod_info

    def get_sample_subject_memmap_no_patch(self, dir_fod_sample):

        lardi_wm = MemMap.read_mem_map(f"{dir_fod_sample}/data_lardi.dat")
        hardi_wm = MemMap.read_mem_map(f"{dir_fod_sample}/data_hardi.dat")

        fodlr_3D_patches = lardi_wm
        fodgt_3D_patches = hardi_wm

        fod_info = {
            'fodlr_3D_patches': fodlr_3D_patches,
            'fodgt_3D_patches': fodgt_3D_patches,
        }
        return fod_info


def pad_vector(vector, desired_shape):
    """
    Preenche o vetor com zeros para que ele tenha o tamanho desejado.

    Args:
        vector (torch.Tensor): O vetor original.
        desired_shape (tuple): O tamanho desejado do vetor.

    Returns:
        torch.Tensor: O vetor preenchido com zeros.
    """
    pad_dims = [max(0, desired - actual) for desired, actual in zip(desired_shape, vector.shape)]
    padded_vector = np.pad(vector, [(0, pad_dim) for pad_dim in pad_dims], mode='constant', constant_values=0)
    return padded_vector


def coordinate_centroid(shape):
    coordinate_centroid = tuple(dim // 2 for dim in shape)
    return coordinate_centroid
