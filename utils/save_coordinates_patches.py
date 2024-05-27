from extractor_patches import ExtractorPatches
from interpolate.five_ttgen import InterpResampleImg
from load_mem_map import MemMap
from create_path import create_paste, create_just_one_paste
import os, torch
import numpy as np
from tqdm import tqdm
from masks.mask_tract_seg import MaskTractSeg
from masks.get_index_patch_mask import SelectPatchWithMask

if __name__ == "__main__":
    abs_path = "insert_your_path_"
    new_path = "/mnt/datahdd/dataset_organized/coordinates_train"
    ids = os.listdir(f"{abs_path}/")
    path_ids = [f"{abs_path}/{x}" for x in ids]
    "data_fsl5ttgen.dat"

    interp_resampling_img = InterpResampleImg()
    extractor_patches = ExtractorPatches((145, 174, 145), tamanho_janela=(96, 96, 96))
    select_patch_with_mask = SelectPatchWithMask()
    class_mask_5ttgen = MaskTractSeg()
    dict_coeff_coordinate = {}
    count = 0

    for idx, path_id in enumerate(path_ids):
        data_fsl5ttgen_shape = MemMap.read_mem_map(f"{path_id}/data_fsl5ttgen_shape.dat",
                                                   data_dtype=np.dtype('int64'),
                                                   data_shape=(4,))

        mask_ = MemMap.read_mem_map(f"{path_id}/data_fsl5ttgen.dat", data_dtype=np.dtype("float64"),
                                    data_shape=tuple(data_fsl5ttgen_shape))
        # data_fsl5ttgen_shape

        mask_affine = MemMap.read_mem_map(f"{path_id}/data_fsl5ttgen_affine.dat",
                                          data_dtype=np.dtype("float64"),
                                          data_shape=(4, 4))
        lardi = MemMap.read_mem_map(f"{path_id}/data_lardi.dat",
                                    data_dtype=np.dtype("float64"),
                                    data_shape=(145, 174, 145, 45))

        lardi_affine = MemMap.read_mem_map(f"{path_id}/data_lardi_affine.dat",
                                           data_dtype=np.dtype("float64"),
                                           data_shape=(4, 4))

        mask5ttgen = interp_resampling_img.interpolate(mask_,
                                                       lardi,
                                                       lardi_affine,
                                                       mask_affine)

        mask5ttgen = np.clip(mask5ttgen, a_min=0.0, a_max=1.0)

        mask_5ttgen = np.where((mask5ttgen > 0), True, False)

        mask_5ttgen = torch.tensor(mask_5ttgen)

        id__ = path_id.replace(f"{abs_path}/", "")

        create_just_one_paste(f"{new_path}/{id__}/coordinates")

        mask_unique = class_mask_5ttgen.apply_union_5ttgen(
            mask_5ttgen[..., 0], mask_5ttgen[..., 1], mask_5ttgen[..., 2]).numpy()

        qtd_patches = extractor_patches.quantidade_de_patches()

        patches = extractor_patches.extrair_patches(mask_unique)

        threshold = 0.15

        for mask in tqdm(patches, total=qtd_patches):
            patch, coordinate = mask
            coeff = select_patch_with_mask.calculate_purity(patch)

            if coeff < threshold:
                continue

            MemMap.write_mem_map(np.array(coordinate), f"{new_path}/{id__}/coordinates/{count}.dat")  # (6,)
            count += 1

        print(f"{ids[idx]}===={count}")

        count = 0
