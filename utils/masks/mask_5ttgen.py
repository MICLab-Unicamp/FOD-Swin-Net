from utils.masks.save_util import save_image_organize
from utils.masks.filter_mask_ttgen import FilterMask5ttgen
from utils.masks.mask_tract_seg import MaskTractSeg
import torch
from utils.interpolate.five_ttgen import InterpResampleImg


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    return device


class Mask5ttgen:
    def __init__(self):
        self.device = set_device()
        self.interpolate_five_ttgen = InterpResampleImg()
        self.filter_mask_5ttgen = FilterMask5ttgen()
        self.mask_tract_seg = MaskTractSeg()
        self.tmp_mask = None

    def apply_mask_5ttgen(self, fodsr, fixed_5ttgen, type_tissue, brain_mask, fod_affine, ttgen_affine):
        # fodsr *= brain_mask.unsqueeze(-1)

        # fodsr = fodsr[5:-5, 5:-5, 5:-5, :]

        if type_tissue == "gray_matter_white_matter":
            fixed_5ttgen_CGM = fixed_5ttgen[..., 0]
            fixed_5ttgen_WM = fixed_5ttgen[..., 2]

            fixed_5ttgen_CGM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_CGM, 0.3)
            fixed_5ttgen_WM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_WM, 0.3)

            CGM_WM = self.mask_tract_seg.apply_intersection_5ttgen(torch.tensor(fixed_5ttgen_CGM),
                                                                   torch.tensor(fixed_5ttgen_WM))

            fixed_5ttgen = self.interpolate_five_ttgen.interpolate(CGM_WM, fodsr, fod_affine,
                                                                   ttgen_affine)

            self.tmp_mask = fixed_5ttgen.copy()

            fixed_5ttgen = torch.tensor(fixed_5ttgen.copy()).to(self.device)

            fodsr *= fixed_5ttgen.unsqueeze(-1)

        elif type_tissue == "subcortical_gray_matter_white_matter":
            fixed_5ttgen_SGM = fixed_5ttgen[..., 1]
            fixed_5ttgen_WM = fixed_5ttgen[..., 2]

            fixed_5ttgen_SGM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_SGM, 0.3)
            fixed_5ttgen_WM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_WM, 0.3)

            SGM_WM = self.mask_tract_seg.apply_intersection_5ttgen(torch.tensor(fixed_5ttgen_SGM),
                                                                   torch.tensor(fixed_5ttgen_WM))

            fixed_5ttgen = self.interpolate_five_ttgen.interpolate(SGM_WM, fodsr, fod_affine,
                                                                   ttgen_affine)

            self.tmp_mask = fixed_5ttgen.copy()

            fixed_5ttgen = torch.tensor(fixed_5ttgen.copy()).to(self.device)

            fodsr *= fixed_5ttgen.unsqueeze(-1)

        elif type_tissue == "apply_filter_train":
            fixed_5ttgen_CGM = fixed_5ttgen[..., 0]
            fixed_5ttgen_SGM = fixed_5ttgen[..., 1]
            fixed_5ttgen_WM = fixed_5ttgen[..., 2]

            fixed_5ttgen_CGM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_CGM, 0.003)
            fixed_5ttgen_SGM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_SGM, 0.003)
            fixed_5ttgen_WM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_WM, 0.003)

            CGM_WM = self.mask_tract_seg.apply_intersection_5ttgen(torch.tensor(fixed_5ttgen_CGM),
                                                                   torch.tensor(fixed_5ttgen_WM))

            three_tissue = self.mask_tract_seg.apply_intersection_5ttgen(torch.tensor(CGM_WM),
                                                                         torch.tensor(fixed_5ttgen_SGM))

            fixed_5ttgen = self.interpolate_five_ttgen.interpolate(three_tissue, fodsr, fod_affine,
                                                                   ttgen_affine)

            self.tmp_mask = fixed_5ttgen.copy()

            fixed_5ttgen = torch.tensor(fixed_5ttgen.copy()).to(self.device)

            fodsr *= fixed_5ttgen.unsqueeze(-1)

        elif type_tissue == "white_matter":
            fixed_5ttgen = fixed_5ttgen[..., 2]

            fixed_5ttgen = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen, 0.7)

            fixed_5ttgen = self.interpolate_five_ttgen.interpolate(fixed_5ttgen, fodsr, fod_affine,
                                                                   ttgen_affine)

            self.tmp_mask = fixed_5ttgen.copy()

            fixed_5ttgen = torch.tensor(fixed_5ttgen.copy()).to(self.device)

            fodsr *= fixed_5ttgen.unsqueeze(-1)

        elif type_tissue == "white_matter_fixel":
            fixed_5ttgen = fixed_5ttgen[..., 2]

            fixed_5ttgen = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen, 0.00001)

            # fixed_5ttgen = self.interpolate_five_ttgen.interpolate(fixed_5ttgen, fodsr, fod_affine,
            #                                                        ttgen_affine)

            self.tmp_mask = fixed_5ttgen.copy()

            fixed_5ttgen = torch.tensor(fixed_5ttgen.copy()).to(self.device)
            try:
                fodsr *= fixed_5ttgen.unsqueeze(-1)
            except:
                print("error")

        return fodsr

    def apply_mask_5ttgen_acc(self, fodsr, fixed_5ttgen, type_tissue, brain_mask, fod_affine, ttgen_affine):
        # fodsr *= brain_mask.unsqueeze(-1)

        # fodsr = fodsr[5:-5, 5:-5, 5:-5, :]

        if type_tissue == "gray_matter_white_matter":
            fixed_5ttgen_CGM = fixed_5ttgen[..., 0]
            fixed_5ttgen_WM = fixed_5ttgen[..., 2]

            fixed_5ttgen_CGM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_CGM, 0.3)
            fixed_5ttgen_WM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_WM, 0.3)

            CGM_WM = self.mask_tract_seg.apply_intersection_5ttgen(torch.tensor(fixed_5ttgen_CGM),
                                                                   torch.tensor(fixed_5ttgen_WM))

            fixed_5ttgen = self.interpolate_five_ttgen.interpolate(CGM_WM, fodsr, fod_affine,
                                                                   ttgen_affine)

            fixed_5ttgen = torch.tensor(fixed_5ttgen.copy()).to(self.device)

            fodsr *= fixed_5ttgen.unsqueeze(-1)

        elif type_tissue == "subcortical_gray_matter_white_matter":
            fixed_5ttgen_SGM = fixed_5ttgen[..., 1]
            fixed_5ttgen_WM = fixed_5ttgen[..., 2]

            fixed_5ttgen_SGM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_SGM, 0.3)
            fixed_5ttgen_WM = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen_WM, 0.3)

            SGM_WM = self.mask_tract_seg.apply_intersection_5ttgen(torch.tensor(fixed_5ttgen_SGM),
                                                                   torch.tensor(fixed_5ttgen_WM))

            fixed_5ttgen = self.interpolate_five_ttgen.interpolate(SGM_WM, fodsr, fod_affine,
                                                                   ttgen_affine)

            fixed_5ttgen = torch.tensor(fixed_5ttgen.copy()).to(self.device)

            fodsr *= fixed_5ttgen.unsqueeze(-1)

        elif type_tissue == "white_matter":
            fixed_5ttgen = fixed_5ttgen[..., 0]
            fixed_5ttgen = fixed_5ttgen[..., 1]
            fixed_5ttgen = fixed_5ttgen[..., 2]

            fixed_5ttgen = self.filter_mask_5ttgen.apply_filter_mask(fixed_5ttgen, 0.7)

            fixed_5ttgen = self.interpolate_five_ttgen.interpolate(fixed_5ttgen, fodsr, fod_affine,
                                                                   ttgen_affine)

            fixed_5ttgen = torch.tensor(fixed_5ttgen.copy()).to(self.device)

            fodsr *= fixed_5ttgen.unsqueeze(-1)

        return fodsr

    def mask_5ttgen(self, fodsr, fixed_5ttgen, type_tissue, brain_mask, fod_affine, ttgen_affine, fod_header, id,
                    save=False, name_path=""):

        fodsr = self.apply_mask_5ttgen(fodsr, fixed_5ttgen, type_tissue, brain_mask, fod_affine, ttgen_affine)

        fodsr = fodsr.detach().cpu().numpy()
        if save:
            save_image_organize(fodsr, id, type_tissue, fod_affine, fod_header, name_path=name_path)
        else:
            ...

        return fodsr, id
