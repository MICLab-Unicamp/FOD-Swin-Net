import os
import random
from typing import List


class SplitDataset:
    def __init__(self):
        ...

    @staticmethod
    def open_txt_samples(dir_data: str) -> List[str]:
        with open(dir_data, 'r') as f:
            read_content = f.read()

        list_generate_samples = read_content.split('\n')
        return list_generate_samples

    @staticmethod
    def write_txt_samples(samples: list, dir_data: str) -> None:
        with open(dir_data, 'w') as f:
            f.write('\n'.join(samples))
        return

    def path_dataset(self, **kargs: dict) -> tuple[List[str], List[str]]:
        dataset_dir = kargs['path']

        split_train = kargs['train']
        split_valid = kargs['valid']
        split_test = kargs['test']

        seed_split = kargs['seeds']

        elements_in_dataset = os.listdir(dataset_dir)

        random.seed(seed_split)

        len_elements_in_dataset = len(elements_in_dataset)

        if split_test + split_valid + split_train != len_elements_in_dataset:
            raise(AttributeError, "problem with test, valid e train split")

        n_rands = random.sample(range(0, len_elements_in_dataset), len_elements_in_dataset)

        idx_samples_valid = n_rands[:split_valid]
        idx_samples_test = n_rands[split_valid:split_test + split_valid]
        idx_samples_train = n_rands[split_test + split_valid:split_train + split_test + split_valid]

        samples_train = [f"{dataset_dir}/{elements_in_dataset[idx]}" for idx in idx_samples_train]
        samples_test = [f"{dataset_dir}/{elements_in_dataset[idx]}" for idx in idx_samples_test]
        samples_valid = [f"{dataset_dir}/{elements_in_dataset[idx]}" for idx in idx_samples_valid]

        print(f"length train: {len(samples_train)}")
        print(f"length test: {len(samples_test)}")
        print(f"length validation: {len(samples_valid)}")
        print(f"Random seed: {seed_split}")

        try:
            os.mkdir(f"{kargs['output_dir']}/{kargs['path_name_create']}")
        except:
            print("path_split already exists")

        self.write_txt_samples(samples_train, f"{kargs['output_dir']}/{kargs['path_name_create']}/train.txt")
        self.write_txt_samples(samples_valid, f"{kargs['output_dir']}/{kargs['path_name_create']}/valid.txt")
        self.write_txt_samples(samples_test, f"{kargs['output_dir']}/{kargs['path_name_create']}/test.txt")

        return samples_train, samples_valid
