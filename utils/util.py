import torch
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
import re

from omegaconf import DictConfig, OmegaConf


def convert_to_standard_dict(omegaconf_dict):
    if isinstance(omegaconf_dict, DictConfig):
        standard_dict = OmegaConf.to_container(omegaconf_dict, resolve=True)
        for key, value in standard_dict.items():
            if isinstance(value, DictConfig):
                standard_dict[key] = convert_to_standard_dict(value)
        return standard_dict
    else:
        # Se o valor não é um DictConfig, apenas retorne o valor
        return omegaconf_dict


def split_path_and_file(path):
    match = re.match(r'^(.*/)([^/]+)$', path)
    if match:
        path_dir, name_file = match.groups()
        return path_dir, name_file
    else:
        return None, None


def flatten_dict_hydra(dict_hydra):
    list_dict__ = [value for key, value in dict_hydra.items()]
    dict__ = {}
    for dict_ in list_dict__:
        for key_ in dict_.keys():
            dict__[key_] = dict_[key_]

    return OmegaConf.create(dict__)


def read_yaml(file: str) -> yaml.loader.FullLoader:
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    return device
