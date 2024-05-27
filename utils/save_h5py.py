import os
import h5py
import shutil
import numpy as np
import nibabel as nib
from tqdm import tqdm
from read_dataset import ReadDataset

read_dataset = ReadDataset(9)


def flip_using_affine(data, affine):
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


def process_data(input_folder, output_folder):
    input_folder_name = os.path.basename(input_folder)
    with tqdm(total=len(os.listdir(input_folder)), desc=f"Processing {input_folder_name}") as pbar_input:
        for root, _, files in os.walk(input_folder):
            with tqdm(total=len(os.listdir(root)), desc="Processing ID") as pbar:
                for id_folder in filter(lambda d: d.isnumeric(), os.listdir(root)):
                    try:
                        id_path = os.path.join(root, id_folder)

                        # Caminhos para os arquivos de interesse
                        wm_fod_path = os.path.join(id_path, 'HARDI_data', 'WM_FODs_normalised.nii.gz')
                        lardi_wm_fod_path = os.path.join(id_path, 'LARDI_data', 'normalized_WM_FODs.nii.gz')
                        t1_5ttgen_path = os.path.join(id_path, 'T1_fsl_5ttgen.nii.gz')

                        # Cria o caminho para a pasta de saída
                        output_id_folder = os.path.join(output_folder, input_folder_name, id_folder)
                        os.makedirs(output_id_folder, exist_ok=True)

                        lardi_wm = nib.load(wm_fod_path)
                        hardi_wm = nib.load(lardi_wm_fod_path)
                        fsl5ttgen_mask = nib.load(t1_5ttgen_path)

                        (fodlr,
                         fixed_fodlr_affine,
                         flipped_axis_fodlr) = read_dataset.flip_axis_to_match_HCP_space(lardi_wm.get_fdata(),
                                                                                         lardi_wm.affine)

                        (hardi_wm,
                         fixed_fodgt_affine,
                         flipped_axis_fodgt) = read_dataset.flip_axis_to_match_HCP_space(hardi_wm.get_fdata(),
                                                                                         hardi_wm.affine)

                        (fsl5ttgen_mask,
                         fixed_fsl5ttgen_mask_affine,
                         flipped_axis_fsl5ttgen_mask) = read_dataset.flip_axis_to_match_HCP_space(
                            fsl5ttgen_mask.get_fdata(),
                            fsl5ttgen_mask.affine)

                        # Cria o arquivo h5py e salva os dados
                        with h5py.File(f"{output_id_folder}/data.hdf5", "w") as h5_file:
                            data, affine = fodlr, fixed_fodlr_affine
                            h5_file.create_dataset("lardi_wm", data=data)
                            h5_file.create_dataset("lardi_wm_affine", data=affine)

                            data, affine = hardi_wm, fixed_fodgt_affine
                            h5_file.create_dataset("hardi_wm", data=data)
                            h5_file.create_dataset("hardi_wm_affine", data=affine)

                            data, affine = fsl5ttgen_mask, fixed_fsl5ttgen_mask_affine
                            h5_file.create_dataset("5ttgen", data=data)
                            h5_file.create_dataset("5ttgen_affine", data=affine)
                    except:
                        print(f"problem with -----{id_path}-----")
                    pbar.update(1)
                pbar_input.update(1)


if __name__ == "__main__":
    input_folders = [
        "/mnt/datahdd/dataset_organized/test",
        "/mnt/datahdd/dataset_organized/train",
        "/mnt/datahdd/dataset_organized/valid"
    ]
    output_folder = "/home/mateus/Documentos/dataset_pos_process_m_oliveira"

    for input_folder in input_folders:
        process_data(input_folder, output_folder)
