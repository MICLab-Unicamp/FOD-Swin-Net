import numpy as np


class ExtractorPatches:
    def __init__(self, imagem_shape, tamanho_janela=(128, 128, 128), stride=1):
        self.imagem_shape = imagem_shape
        self.tamanho_janela = tamanho_janela
        self.stride = stride

    def quantidade_de_patches(self):
        tamanho_x, tamanho_y, tamanho_z = self.tamanho_janela
        stride_x, stride_y, stride_z = self.stride, self.stride, self.stride

        quantidade_x = (self.imagem_shape[0] - tamanho_x) // stride_x + 1
        quantidade_y = (self.imagem_shape[1] - tamanho_y) // stride_y + 1
        quantidade_z = (self.imagem_shape[2] - tamanho_z) // stride_z + 1

        return quantidade_x * quantidade_y * quantidade_z

    def extrair_patches(self, imagem):
        tamanho_x, tamanho_y, tamanho_z = self.tamanho_janela
        stride_x, stride_y, stride_z = self.stride, self.stride, self.stride

        for x in range(0, self.imagem_shape[0] - tamanho_x + 1, stride_x):
            for y in range(0, self.imagem_shape[1] - tamanho_y + 1, stride_y):
                for z in range(0, self.imagem_shape[2] - tamanho_z + 1, stride_z):
                    patch = imagem[x:x + tamanho_x, y:y + tamanho_y, z:z + tamanho_z]

                    coordinate = (x, x + tamanho_x, y, y + tamanho_y, z, z + tamanho_z)

                    yield patch, coordinate


if __name__ == "__main__":
    imagem_exemplo = np.random.rand(145, 174, 145)
    extrator = ExtractorPatches(imagem_shape=imagem_exemplo.shape,
                                tamanho_janela=(64, 64, 64),
                                stride=1)
