import nibabel as nib


def filter_values(img, low, high):
    values_mask = (img > low) & (img < high)
    select_values = img[values_mask]
    return select_values


img = nib.load('/mnt/datahdd/dataset_organized/train/100206/fsl_5ttgen.nii.gz')

img = img.get_fdata()

img1 = img[..., 1]
select_values = filter_values(img1, 0.1, 0.9)
print(f"img Sub cortical gray matter:{select_values.shape}")

img1 = img[..., 0]
select_values = filter_values(img1, 0.1, 0.9)
print(f"img cortical gray matter:{select_values.shape}")

img1 = img[..., 2]
select_values = filter_values(img1, 0.1, 0.9)
print(f"img White matter:{select_values.shape}")
