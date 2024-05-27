import pandas as pd
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from skimage.filters import gaussian
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize
from dipy.io.image import load_nifti


class PlotRawDegree:
    def __init__(self, dataframes_list: List[pd.DataFrame]):
        self.dataframes_list = dataframes_list

        self.x_label = "Ângulo (graus)"
        self.y_label = "Porcentagem (%)"

        self.x_range = (0, 90)
        self.y_range = (0, 100)

        self.colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#1e2833", "#FF5733", "#007ACC", "#00CC00", "#FF00FF",
            "#FFFF00", "#FFA500", "#800080", "#00FFFF", "#FF1493"
        ]

        self.names = ['5e-07_batch_1000', '0.0005_batch_100', '0.005_batch_1000',
                      '0.005_batch_5000', '5e-07_batch_5000', '5e-06_batch_5000',
                      '5e-06_batch_256', '5e-06_batch_100', '5e-07_batch_100',
                      '0.0005_batch_5000', '0.0005_batch_1000', '0.005_batch_256',
                      '5e-06_batch_1000', '0.005_batch_100', '5e-07_batch_256']

    def plot_raw_degree(self, name_file: str):
        plt.figure(figsize=(16, 3))

        for i, df in enumerate(self.dataframes_list):
            mean_angular_error = df["MAE"].tolist()
            sorted_mean_angular_error = sorted(mean_angular_error)
            length = len(sorted_mean_angular_error)
            list_y = [x / (length - 1) * 100 for x in range(length)]  # Escala para 0-100%

            # Plot com cores diferentes
            color = self.colors[i % len(self.colors)]
            plt.plot(sorted_mean_angular_error, list_y, label=f'{self.names[i]}', color=color)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title("Ângulo x Porcentagem")
        plt.xticks(np.arange(self.x_range[0], self.x_range[1] + 1, 10))
        plt.yticks(np.arange(self.y_range[0], self.y_range[1] + 1, 10))
        plt.xlim(self.x_range)
        plt.ylim(self.y_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{name_file}.png', dpi=300)


class MiddleSliceDegreePlot:
    def __init__(self): ...

    def slice_degree_visual_metric_acc(self, result_angles: torch.Tensor, name_image: str, mask_path):
        # name_image = name_image.replace("/mnt/datahdd/dataset_organized/train/", "")

        # result_graus_angle = torch.rad2deg(torch.acos(result_angles))
        # img1 = abs(90 - np.rot90(result_graus_angle[:, :, result_graus_angle.shape[2] // 2].numpy()))

        mask, _ = load_nifti(mask_path)

        mask_wm = np.zeros(mask[..., 0].shape, "uint8")
        mask_wm[mask[..., 2] > 0.0001] = 1

        # Definir uma sequência de cores
        colors = [
            (0, 0, 0),  # Preto
            (0, 0, 1),  # Azul
            (0, 1, 1),  # Ciano
            (1, 1, 1)  # Branco
        ]

        colors.reverse()

        locations = [0, 0.5, 0.9, 1]

        cmap_name = "custom_adjusted_cmap"
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(locations, colors)))

        result_graus_angle = result_angles # np.degrees(np.arccos(np.abs(result_angles)))
        img1 = np.rot90(result_graus_angle[:, :, result_graus_angle.shape[2] // 2])
        # Usar a colormap customizada no lugar de "hot"

        # img2 = gaussian(img1, 0.7)
        new_shape = (img1.shape[0] * 7, img1.shape[1] * 7)

        mask_wm = np.rot90(mask_wm[:, :, result_graus_angle.shape[2] // 2])
        mask_wm = np.flip(mask_wm, axis=1)

        # mask_wm_inverted = 1 - mask_wm
        # mask_wm_final = mask_wm_inverted * 90

        # img1 = img1 * mask_wm
        # Interpolação bilinear
        img2 = resize(img1, new_shape, mode='reflect', anti_aliasing=True)
        # mask2 = resize(np.rot90(mask_wm[:, :, result_graus_angle.shape[2] // 2]),
        #               new_shape, mode='reflect',
        #               anti_aliasing=True)

        plt.imshow(img2, interpolation='nearest', cmap=custom_cmap)

        cbar = plt.colorbar()
        #cbar.set_label('', rotation=0, labelpad=10)
        # cbar.set_ticks(np.arange(-1, 1, 10))
        #cbar.set_ticklabels(np.arange(90, -1, -10))
        plt.xticks([])
        plt.yticks([])

        plt.savefig(f'{name_image}_acc_.png', dpi=300)

        plt.clf()
    def slice_degree_visual_metric(self, result_angles: torch.Tensor, name_image: str, mask_path):
        # name_image = name_image.replace("/mnt/datahdd/dataset_organized/train/", "")

        # result_graus_angle = torch.rad2deg(torch.acos(result_angles))
        # img1 = abs(90 - np.rot90(result_graus_angle[:, :, result_graus_angle.shape[2] // 2].numpy()))

        mask, _ = load_nifti(mask_path)

        mask_wm = np.zeros(mask[..., 0].shape, "uint8")
        mask_wm[mask[..., 2] > 0.0001] = 1

        # Definir uma sequência de cores
        colors = [
            (0, 0, 0),  # Preto
            (0, 0, 1),  # Azul
            (0, 1, 1),  # Ciano
            (1, 1, 1)  # Branco
        ]

        locations = [0, 0.1, 0.7, 1]

        cmap_name = "custom_adjusted_cmap"
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(locations, colors)))

        result_graus_angle = np.degrees(np.arccos(np.abs(result_angles)))
        img1 = np.abs(np.rot90(result_graus_angle[:, :, result_graus_angle.shape[2] // 2]))
        # Usar a colormap customizada no lugar de "hot"

        # img2 = gaussian(img1, 0.7)
        new_shape = (img1.shape[0] * 7, img1.shape[1] * 7)

        mask_wm = np.rot90(mask_wm[:, :, result_graus_angle.shape[2] // 2])
        mask_wm = np.flip(mask_wm, axis=1)

        # mask_wm_inverted = 1 - mask_wm
        # mask_wm_final = mask_wm_inverted * 90

        img1 = img1 * mask_wm
        # Interpolação bilinear
        img2 = resize(img1, new_shape, mode='reflect', anti_aliasing=True)
        # mask2 = resize(np.rot90(mask_wm[:, :, result_graus_angle.shape[2] // 2]),
        #               new_shape, mode='reflect',
        #               anti_aliasing=True)

        plt.imshow(img2, interpolation='nearest', cmap=custom_cmap)

        cbar = plt.colorbar()
        #cbar.set_label('', rotation=0, labelpad=10)
        #cbar.set_ticks(np.arange(0, 91, 10))
        #cbar.set_ticklabels(np.arange(90, -1, -10))
        plt.xticks([])
        plt.yticks([])

        plt.savefig(f'{name_image}.png', dpi=300)

        plt.clf()
