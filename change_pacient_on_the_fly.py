import os
from tqdm import tqdm
import gc


class ChangePacientOnTheFly:
    def __init__(self):
        self.idx_patch = 0

    def apply(self, loader, read_dataset, pacient):
        data_list_id = loader.dataset.data_list_id
        absolute_path = loader.dataset.absolute_path

        len_dataset = len(loader.dataset.data_list_id)

        try:
            element = read_dataset.load_hcp(absolute_path, data_list_id[pacient % len_dataset])
            if element is not None:
                loader.dataset.element = element
            else:
                return loader
        except:
            print(f"---- ERROR --- {data_list_id[pacient % len_dataset]}")
            return None

        print(f"---------> pacient {data_list_id[pacient % len_dataset]}")
        return loader

    def __pop_id_elements(self, count, data_list_id):
        get_elements = []
        for _ in range(count):
            if data_list_id == []:
                break
            get_elements.append(data_list_id.pop())

        return get_elements, data_list_id

    def __reajust_len_dataset(self, loader, count_subject):
        len_index_ids_patch = loader.dataset.element[0]["index_mask"].shape[1]

        loader.dataset.dataset_num_samples_per_data = count_subject * len_index_ids_patch

        return loader

    def apply_many_subjects(self, loader, read_dataset):
        loader.dataset.element = []
        data_list_id_backup = loader.dataset.data_list_id_backup
        data_list_id = loader.dataset.data_list_id
        absolute_path = loader.dataset.absolute_path
        count_subjects = loader.dataset.count_subjects

        subjects, data_list_id = self.__pop_id_elements(count_subjects, data_list_id)

        if len(subjects) < count_subjects:
            loader.dataset.data_list_id = data_list_id_backup.copy()
            data_list_id = loader.dataset.data_list_id
            subjects, data_list_id = self.__pop_id_elements(count_subjects, data_list_id)

        for idx, subject in enumerate(tqdm(subjects, desc="Loading subjects")):

            readed_subject = read_dataset.load_hcp_list(absolute_path, subject)

            if readed_subject[0] is None:
                print(f"---- ERROR --- {subject}")
                readed_subject += loader.dataset.element[0] # pega o primeiro elemento para add novamente
                continue

            loader.dataset.element += readed_subject

        # loader = self.__reajust_len_dataset(loader, count_subjects)

        print(f"---------> pacient {subjects}")

        return loader

    def patches_on_the_fly(self, loader, read_dataset):
        data_list_id = loader.dataset.data_list_id
        absolute_path = loader.dataset.absolute_path

        data_patch_lis = []

        for i, subject_id in tqdm(enumerate(data_list_id)):
            subject_sample = read_dataset.add_fod_sample(fodlr_path=os.path.join(absolute_path,
                                                                                 subject_id,
                                                                                 "HARDI_data",
                                                                                 "WM_FODs_normalised.nii.gz"),
                                                         fodgt_path=os.path.join(absolute_path, subject_id,
                                                                                 "LARDI_data",
                                                                                 "normalized_WM_FODs.nii.gz"),
                                                         fsl_5ttgen_mask_path=os.path.join(absolute_path,
                                                                                           subject_id,
                                                                                           "T1_fsl_5ttgen.nii.gz"),
                                                         subject_id=subject_id)

            (fodlr_3D_patches,
             fodgt_3D_patches,
             self.idx_patch) = read_dataset.just_one_cube(subject_sample, self.idx_patch)

            data_patch_lis.append({"fodlr": fodlr_3D_patches,
                                   "fodgt": fodgt_3D_patches})

        loader.dataset.elements = data_patch_lis

        return loader
