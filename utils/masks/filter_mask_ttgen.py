import numpy as np
import torch


class FilterMask5ttgen:
    def __init__(self):
        ...

    def apply_filter_mask(self, mask, threshold):
        if isinstance(mask, np.ndarray):
            # Para NumPy arrays
            mask = np.where((mask > threshold), mask, 0)
            mask = np.where((mask > 0), 1, 0)
        else:
            raise ValueError("Value not supported")

        return mask
