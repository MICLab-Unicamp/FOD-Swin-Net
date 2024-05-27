import torch
import random
import numpy as np


class ImagePad:
    def __init__(self): ...

    def split_and_yield_random(self, image_1, image_2, padding_size_x_z=11, padding_size_y=7):
        padded_image_1 = np.zeros((192, 192, 192, 45))
        padded_image_1[padding_size_x_z:padding_size_x_z + image_1.shape[0],
        padding_size_y:padding_size_y + image_1.shape[1],
        padding_size_x_z:padding_size_x_z + image_1.shape[2], :] = image_1

        padded_image_2 = np.zeros((192, 192, 192, 45))
        padded_image_2[padding_size_x_z:padding_size_x_z + image_2.shape[0],
        padding_size_y:padding_size_y + image_2.shape[1],
        padding_size_x_z:padding_size_x_z + image_2.shape[2], :] = image_2

        height, width, depth, _ = (192, 192, 192, 45)
        quarter_height = height // 2
        quarter_width = width // 2
        quarter_depth = depth // 2

        splits_1 = [
            padded_image_1[:quarter_height, quarter_width:, :quarter_depth, :],
            padded_image_1[:quarter_height, quarter_width:, :quarter_depth, :],
            padded_image_1[quarter_height:, :quarter_width, :quarter_depth, :],
            padded_image_1[quarter_height:, quarter_width:, :quarter_depth, :]
        ]
        splits_2 = [
            padded_image_2[:quarter_height, quarter_width:, :quarter_depth, :],
            padded_image_2[:quarter_height, quarter_width:, :quarter_depth, :],
            padded_image_2[quarter_height:, :quarter_width, :quarter_depth, :],
            padded_image_2[quarter_height:, quarter_width:, :quarter_depth, :]
        ]

        selected_split = random.randint(0, 3)  # Escolhe aleatoriamente um dos quatro splits

        return splits_1[selected_split], splits_2[selected_split]

    def pad_zeros(self, image, padding_size_x_z=11, padding_size_y=7):
        image = image.detach().cpu().numpy()

        padded_image_1 = np.zeros((45, 192, 192, 192))
        padded_image_1[:, padding_size_x_z:padding_size_x_z + image.shape[1],
        padding_size_y:padding_size_y + image.shape[2],
        padding_size_x_z:padding_size_x_z + image.shape[3]] = image

        _, height, width, depth = (45, 192, 192, 192)
        quarter_height = height // 2
        quarter_width = width // 2
        quarter_depth = depth // 2

        splits_1 = [
            padded_image_1[:, :quarter_height, quarter_width:, :quarter_depth],
            padded_image_1[:, :quarter_height, quarter_width:, :quarter_depth],
            padded_image_1[:, quarter_height:, :quarter_width, :quarter_depth],
            padded_image_1[:, quarter_height:, quarter_width:, :quarter_depth]
        ]

        splits_1 = [torch.tensor(pad, dtype=torch.float32).to("cuda") for pad in splits_1]

        return splits_1

    def reconstruction_image(self, splits_1, padding_size_x_z=11, padding_size_y=7):
        # Verifica o tamanho de um dos splits
        _, split_height, split_width, split_depth = splits_1[0].shape

        # Calcula o tamanho original da imagem
        original_shape = (45, 145, 174, 145)

        # Calcula o tamanho do preenchimento em cada dimensão
        padding = [original_shape[i] - splits_1[0].shape[i] for i in range(len(original_shape))]

        # Divide o padding pela metade, pois foi adicionada metade em cada extremidade
        padding = [p // 2 for p in padding]

        # Calcula os índices de fatia para recuperar as partes originais
        slice_indices = [slice(padding_size_x_z, padding_size_x_z + split_height),
                         slice(padding_size_y, padding_size_y + split_width),
                         slice(padding_size_x_z, padding_size_x_z + split_depth)]

        # Inicializa uma imagem vazia com o tamanho original
        original_image = np.zeros(original_shape)

        # Recupera as partes originais e as coloca na imagem original
        original_image[:, slice_indices[0], slice_indices[1], slice_indices[2]] = splits_1[0].detach().cpu().numpy()
        original_image[:, slice_indices[0], slice_indices[1], slice_indices[2]] += splits_1[1].detach().cpu().numpy()
        original_image[:, slice_indices[0], slice_indices[1], slice_indices[2]] += splits_1[2].detach().cpu().numpy()
        original_image[:, slice_indices[0], slice_indices[1], slice_indices[2]] += splits_1[3].detach().cpu().numpy()

        # Converte a imagem resultante para um tensor do PyTorch
        original_image_tensor = torch.tensor(original_image, dtype=torch.float32).to("cuda")

        return original_image_tensor

    def remove_zeros(self, image, padding_size_x_z=23, padding_size_y=14):
        cropped_image = image[:, padding_size_x_z:image.shape[1] - padding_size_x_z,
                        padding_size_y:image.shape[2] - padding_size_y,
                        padding_size_x_z:image.shape[3] - padding_size_x_z]

        return cropped_image
