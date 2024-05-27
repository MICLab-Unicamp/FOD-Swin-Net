import numpy as np


class SelectPatchWithMask:
    def __init__(self):
        self.coordinate = None

    def crop_mask(self, mask, patch_shape):
        x_center = (mask.shape[0] - patch_shape[0]) // 2
        y_center = (mask.shape[1] - patch_shape[1]) // 2
        z_center = (mask.shape[2] - patch_shape[2]) // 2

        x_start = x_center - patch_shape[0] // 2
        y_start = y_center - patch_shape[1] // 2
        z_start = z_center - patch_shape[2] // 2

        x_end = x_start + patch_shape[0]
        y_end = y_start + patch_shape[1]
        z_end = z_start + patch_shape[2]

        self.coordinate = (x_start, x_end, y_start, y_end, z_start, z_end)

        return mask[x_start:x_end, y_start:y_end, z_start:z_end]

    def calculate_impurity(self, mask):
        total_elements = mask.size
        true_elements = np.count_nonzero(mask)
        impurity = 1.0 - (true_elements / total_elements)
        return impurity

    def calculate_purity(self, mask):
        total_elements = mask.size
        true_elements = np.count_nonzero(mask)
        impurity = (true_elements / total_elements)
        return impurity

    def random_indices(self, mask):
        indices = np.argwhere(mask)
        # Desconsiderar a primeira dimensão
        # indices = indices[:, 1:]
        return indices

    def get_patch(self, mask, patch_shape):
        indices = self.random_indices(mask)
        cropped_mask = mask
        rand_index = np.random.randint(len(indices))
        x, y, z = indices[rand_index]

        while cropped_mask.shape != (64, 64, 64):
            x_start = max(x - patch_shape[0] // 2, 0)
            y_start = max(y - patch_shape[1] // 2, 0)
            z_start = max(z - patch_shape[2] // 2, 0)

            x_end = min(x_start + patch_shape[0], mask.shape[0])
            y_end = min(y_start + patch_shape[1], mask.shape[1])
            z_end = min(z_start + patch_shape[2], mask.shape[2])

            cropped_mask = mask[x_start:x_end, y_start:y_end, z_start:z_end]

        impurity = self.calculate_impurity(cropped_mask)

        return cropped_mask, (x_start, x_end, y_start, y_end, z_start, z_end), impurity

#
# # Exemplo de uso
# selector = SelectPatchWithMask()
# mask_shape = (45, 174, 145, 174)
# patch_shape = (9, 9, 9)
# cropped_mask, coordenadas, impurity = selector.get_patch(mask_shape, patch_shape)
#
# print("Máscara recortada:")
# print(cropped_mask)
# print("\nÍndices dos elementos True:")
# print(coordenadas)
# print("\nGrau de impureza:")
# print(impurity)
