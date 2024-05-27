import nibabel as nib
import torch
import numpy as np
from nilearn.image import resample_to_img


class InterpResampleImg:
    def __init__(self):
        ...

    def interpolate(self, mask, image, affine, ttgen_affine) -> np.ndarray:

        mask = self.torch_to_nibabel(mask, ttgen_affine)
        image = self.torch_to_nibabel(image, affine)

        mask = resample_to_img(mask, image)
        return mask.get_fdata()# .astype(image.dtype)

    @staticmethod
    def save_interp_for_view(mask: np.ndarray, image: np.ndarray, name: str = 'new_mask.nii.gz') -> None:
        imagem_mascarada_nii = nib.Nifti1Image(mask, affine=image.affine)
        nib.save(imagem_mascarada_nii, name)

    def torch_to_nibabel(self, tensor, affine):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().detach().numpy()

        nifti_image = nib.Nifti1Image(tensor.astype(np.float32),
                                      affine=affine)

        return nifti_image
