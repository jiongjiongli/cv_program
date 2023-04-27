from pathlib import Path
import logging
from tqdm import tqdm
import random
import yaml
import shutil
import os


current_dir_path = Path(__file__).resolve().parent

def get_dir_path_dict_collection():
    dir_path_dict_collection = {}
    # cvmart
    dir_path_dict_collection['cvmart'] = {
        'code'         : '/project/train/src_repo/ultralytics',
        'trained_model': '/project/train/models',
        'log'          : '/project/train/log/log.txt',
        'tensorboard'  : '/project/train/tensorboard',
        'data'         : '/home/data'
    }

    # colab
    dir_path_dict_collection['colab'] = {
        'code'         : '/content/ultralytics',
        'trained_model': '/content/models',
        'log'          : '/content/log/log.txt',
        'tensorboard'  : '/content/tensorboard',
        'data'         : '/content/datasets'
    }

    # local
    dir_path_dict_collection['local'] = {
        'code'         : Path('D:/proj/git/ultralytics'),
        'trained_model': (current_dir_path / 'models'),
        'log'          : (current_dir_path / 'log/log.txt'),
        'tensorboard'  : (current_dir_path / 'tensorboard'),
        'data'         : 'D:/Data/cv/experiments/data'
    }

    for key, path_dict in dir_path_dict_collection.items():
        for path_key, path_value in path_dict.items():
            path_dict[path_key] = Path(path_value)

    return dir_path_dict_collection


dir_path_dict_collection = get_dir_path_dict_collection()
