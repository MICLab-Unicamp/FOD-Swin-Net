import numpy as np


class MemMap:
    @staticmethod
    def write_mem_map(data, memmap_filename):
        memmap_array = np.memmap(memmap_filename,
                                 dtype=data.dtype,
                                 mode="w+",
                                 shape=data.shape)

        memmap_array[:] = data[:]

        del memmap_array

        return data.dtype

    @staticmethod
    def read_mem_map(memmap_filename, data_dtype=np.dtype("float64"), data_shape=(145, 174, 145, 45)):
        memmap_data = np.memmap(memmap_filename, dtype=data_dtype, mode='r', shape=data_shape)
        return memmap_data
