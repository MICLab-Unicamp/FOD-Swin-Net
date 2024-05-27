import os


def create_paste(path_name, idx):
    if not os.path.exists(f"{path_name[idx]}"):
        os.makedirs(f"{path_name[idx]}")
    else:
        ...


def create_just_one_paste(path_name):
    if not os.path.exists(f"{path_name}"):
        os.makedirs(f"{path_name}")
        os.chmod(path_name, 0o777)
    else:
        ...
