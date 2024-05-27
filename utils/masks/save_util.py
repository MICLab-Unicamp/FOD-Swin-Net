import nibabel as nib
import os


def save_image_organize(fodsr, id, type_tissue, fod_affine, fod_header, name_path=""):
    nii = nib.Nifti1Image(
        fodsr, affine=fod_affine, header=fod_header)

    id = id.replace("/mnt/datahdd/dataset_organized/train/", "")
    # /mnt/datahdd/dataset_organized/swin_tmp_reconstruction_predicts
    if not os.path.exists(f"{name_path}/{id}"):
        os.makedirs(f"{name_path}/{id}")
    else:
        ...

    nib.save(nii, f"{name_path}/{id}/{type_tissue}_{id}.nii.gz")


def save_image_organize_seg_train(fodsr, id, type_tissue, fod_affine, fod_header, name_path=""):
    nii = nib.Nifti1Image(
        fodsr, affine=fod_affine, header=fod_header)


    nib.save(nii,
             f"{type_tissue}.nii.gz")


def save_image_organize_seg(fodsr, id, type_tissue, fod_affine, fod_header, name_path=""):
    nii = nib.Nifti1Image(
        fodsr, affine=fod_affine, header=fod_header)


    nib.save(nii,
              f"{name_path}/{type_tissue}.nii.gz")


def save_image_(fodsr, id, type_tissue, fod_affine, fod_header, name_path=""):
    nii = nib.Nifti1Image(
        fodsr, affine=fod_affine, header=fod_header)


    nib.save(nii, f"{type_tissue}/{id}.nii.gz")