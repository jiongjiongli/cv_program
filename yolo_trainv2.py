from pathlib import Path
import logging
from tqdm import tqdm
import random
import yaml
import shutil
import os
import hashlib
import uuid

from yolo_dir_path_manager import dir_path_dict_collection
from convert_voc_to_yolo import generate_yolo_dataset
from ultralytics.yolo.utils import USER_CONFIG_DIR, SETTINGS
from ultralytics.yolo.utils import LOGGER
from ultralytics import YOLO


def add_log_file_handler(log_file_path):
    file_handler = logging.FileHandler(log_file_path.as_posix())
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def main():
    project_name = r'illegal_construction'
    environment_name = 'cvmart'
    pretrained_model_name = 'yolov8n.pt'

    # Resume training.
    is_resume = True
    last_model_path = '/project/train/models/last.pt'
    last_model_path = Path(last_model_path)

    # Prepare
    dir_path_dict = dir_path_dict_collection[environment_name]
    # 1. For 'yolov8n.pt, download model to dir_path_dict['code'] / 'weights' from
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    # 国内使用
    # !wget https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    pretrained_model_path = dir_path_dict['code'] / 'weights' / pretrained_model_name
    assert pretrained_model_path.exists(), r'Error: pretrained_model_path not exist: {}'.format(pretrained_model_path)

    if is_resume:
        assert last_model_path.exists(), r'Error: last_model_path not exist: {}'.format(last_model_path)

    # 2. Download font file to dir_path_dict['code'] from https://ultralytics.com/assets/Arial.ttf
    # Pretrained model
    font_file_path = dir_path_dict['code'] / 'Arial.ttf'
    assert font_file_path.exists(), r'Error: font_file_path not exist: {}'.format(font_file_path)

    # 3. Data files dir path
    data_dir_path = dir_path_dict['data']

    # 4. Log dir path
    log_dir_path = dir_path_dict['log']

    # 5.trained_model dir path
    train_results_dir = Path(dir_path_dict['trained_model'])  / 'detect'

    # 6. Train args
    train_kwargs = dict(project=train_results_dir,
                        epochs=100,
                        batch=32,
                        imgsz=640,
                        # optimizer='AdamW',
                        resume=is_resume)

    # 7. Update tensorboard logs in file:
    # ultralytics/yolo/utils/callbacks/tensorboard.py

    # Add log dir.
    add_log_file_handler(log_dir_path)

    # Copy pretrained model.
    # amp_allclose(YOLO('yolov8n.pt'), im) in ultralytics/yolo/engine/trainer.py
    pretrained_model_dest_path = Path(SETTINGS['weights_dir']) / pretrained_model_path.name

    if not pretrained_model_dest_path.exists():
        pretrained_model_dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pretrained_model_path.as_posix(),
                        pretrained_model_dest_path.as_posix())

    # Copy font file.
    # check_font in ultralytics/yolo/utils/checks.py
    font_dest_file_path = Path(USER_CONFIG_DIR) / font_file_path.name

    if not font_dest_file_path.exists():
        font_dest_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(font_file_path.as_posix(),
                        font_dest_file_path.as_posix())

    # Dataset
    dataset_yaml_file_path = generate_yolo_dataset(project_name, data_dir_path)

    # Train
    model_path = last_model_path if is_resume else pretrained_model_path
    LOGGER.info('Start train...')
    model = YOLO(model_path)
    results = model.train(data=dataset_yaml_file_path.as_posix(), **train_kwargs)

    LOGGER.info('Completed train.')

if __name__ == '__main__':
    main()
