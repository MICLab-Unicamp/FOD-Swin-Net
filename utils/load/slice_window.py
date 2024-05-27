import numpy as np
import random


class SlidingWindow:
    def __init__(self, image_shape, crop_shape):
        self.image_shape = image_shape
        self.crop_shape = crop_shape

    def calculate_total_crops(self):
        total_crops = [
            (self.image_shape[i] - self.crop_shape[i] + 1)
            for i in range(3)
        ]
        return np.prod(total_crops)

    def calculate_total_crops_tuple(self):
        total_crops = [
            (self.image_shape[i] - self.crop_shape[i] + 1)
            for i in range(3)
        ]
        return tuple(total_crops)

    def generate_crops(self, image):
        for i in range(self.total_crops[0]):
            for j in range(self.total_crops[1]):
                for k in range(self.total_crops[2]):
                    crop = image[i:i + self.crop_shape[0], j:j + self.crop_shape[1], k:k + self.crop_shape[2]]
                    yield crop

    def get_total_possible_crops(self):
        return self.calculate_total_crops()

    def generate_random_crops(self, image):
        total_crops = self.calculate_total_crops_tuple()

        i = random.randint(0, total_crops[0] - 1)
        j = random.randint(0, total_crops[1] - 1)
        k = random.randint(0, total_crops[2] - 1)

        crop = image[i:i + self.crop_shape[0], j:j + self.crop_shape[1], k:k + self.crop_shape[2]]
        yield crop
