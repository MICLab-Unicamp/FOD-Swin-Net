import os
import numpy as np
from read_dataset import ReadDataset
from tqdm import tqdm

read_dataset = ReadDataset(9)

absolut_path = "/home/mateus/Documentos/dataset_pos_process_m_oliveira"

path_train = f"{absolut_path}/train"
path_valid = f"{absolut_path}/valid"
path_test = f"{absolut_path}/test"

path_list_id_train = [f"{path_train}/{id}" for id in os.listdir(path_train)]
path_list_id_valid = [f"{path_valid}/{id}" for id in os.listdir(path_valid)]
path_list_id_test = [f"{path_test}/{id}" for id in os.listdir(path_test)]

for path_id in tqdm(path_list_id_train):
    try:
        fod_info = read_dataset.get_sample_subject(path_id)
        np.save(f"{path_id}/index_wm.npy", fod_info["index_mask"])
    except:
        print(f"problem with {path_id}")

for path_id in tqdm(path_list_id_valid):
    fod_info = read_dataset.get_sample_subject(path_id)
    np.save(f"{path_id}/index_wm.npy", fod_info["index_mask"])

for path_id in tqdm(path_list_id_test):
    fod_info = read_dataset.get_sample_subject(path_id)
    np.save(f"{path_id}/index_wm.npy", fod_info["index_mask"])

# np.save('mask_index.npy', my_array)
# np.load('mask_index.npy')