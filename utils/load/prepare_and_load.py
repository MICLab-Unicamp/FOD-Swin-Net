from utils.masks.mask_tract_seg import MaskTractSeg
from utils.masks.mask_5ttgen import Mask5ttgen
from utils.util import set_device
from predict_tools.tools_for_predict_model import AdjustInputTest
import torch

DEVICE = set_device()


class PrepareLoadEvaluate:
    def __init__(self):
        self.mask_5ttgen = Mask5ttgen()
        self.adjust_input_test = AdjustInputTest(DEVICE)

    def apply(self, test_WM, test_MASK, test_WM_ground_truth, test_5ttgen, id, type_tissue="white_matter"):
        (fixed_fodlr,
         fixed_brain_mask,
         fixed_fodlr_affine,
         fodlr_file,
         fixed_ground_truth,
         fixed_5ttgen,
         fixed_5ttgen_affine) = self.adjust_input_test.get_subject_process(test_WM, test_MASK,
                                                                           test_WM_ground_truth,
                                                                           test_5ttgen)

        self.adjust_input_test.input_for_test(fixed_fodlr,
                                              fixed_brain_mask,
                                              fixed_fodlr_affine,
                                              fodlr_file.header,
                                              fixed_5ttgen_affine)

        fod_ground_truth, _ = self.mask_5ttgen.mask_5ttgen(torch.tensor(fixed_ground_truth).to("cuda"),
                                                           fixed_5ttgen, type_tissue,
                                                           self.adjust_input_test.brain_mask,
                                                           self.adjust_input_test.fod_affine,
                                                           self.adjust_input_test.ttgen_affine,
                                                           self.adjust_input_test.fod_header, id, save=False)

        fixed_fodlr, _ = self.mask_5ttgen.mask_5ttgen(torch.tensor(fixed_fodlr).to("cuda"),
                                                      fixed_5ttgen, type_tissue,
                                                      self.adjust_input_test.brain_mask,
                                                      self.adjust_input_test.fod_affine,
                                                      self.adjust_input_test.ttgen_affine,
                                                      self.adjust_input_test.fod_header, id, save=False)

        return fod_ground_truth, fixed_fodlr, self.mask_5ttgen.tmp_mask
