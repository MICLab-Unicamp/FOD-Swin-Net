import torch
from typing import Dict


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    return device


class MaskTractSeg:
    def __init__(self):
        self.device = set_device()

    def apply_intersection_seg(self, dict_roi_s: Dict) -> torch.Tensor:
        bundle_roi_s = [torch.Tensor(dict_roi_s[bundle]["fixed_bundle"]) for i, bundle in enumerate(dict_roi_s.keys())]
        just_one_mask = self.__union(bundle_roi_s)
        return just_one_mask

    def apply_union_5ttgen(self, *masks):
        union_masks = self.__union(masks)
        return union_masks

    def apply_intersection_5ttgen(self, *masks):
        union_masks = self.__intersection(masks)
        return union_masks

    def __intersection(self, masks):

        if len(masks) <= 1:
            return masks[0]

        unioon = masks[0].clone().int()

        # Itera sobre as máscaras restantes

        for mascara in masks[1:]:

            unioon = unioon & mascara# .int()

        return unioon

    def __union(self, masks):

        if len(masks) <= 1:
            return masks[0]

        unioon = masks[0].clone().int()

        # Itera sobre as máscaras restantes

        for mascara in masks[1:]:
            if mascara.dtype==torch.float32:
                unioon = unioon | mascara.int()
            else:
                unioon = unioon | mascara# .int()

        return unioon

    def __itersection(self, masks):

        if len(masks) <= 1:
            return masks[0]  # just one mask

        intersecao = masks[0].clone().int()

        # Itera sobre as máscaras restantes
        for mascara in masks[1:]:
            intersecao = intersecao & mascara.int()
            # intersecao = intersecao | mascara.int()

        return intersecao

    def type_apply_mask(self, fodsr: torch.Tensor, roi_s: Dict):

        fodsr = torch.Tensor(fodsr).to(self.device)

        just_one_mask_intersection = self.apply_intersection_seg(roi_s)

        just_one_mask_intersection = just_one_mask_intersection.to(self.device)

        fodsr *= just_one_mask_intersection.unsqueeze(-1)

        return fodsr
