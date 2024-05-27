import os
import shutil


def remove_directory(directory):
    """ Remove the specified directory and all its contents. """
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Directory '{directory}' has been removed successfully.")
        else:
            print(f"Directory '{directory}' does not exist.")
    except Exception as e:
        print(f"ocorreu um erro: {e} ao tentar apagar: {directory}")
