from yolo_dir_path_manager import dir_path_dict_collection
from convert_voc_to_yolo import generate_yolo_dataset
from pathlib import Path
import logging
from tqdm import tqdm
import random
import yaml
import shutil
import os
import hashlib
import uuid


def add_log_file_handler(log_file_path):
    from ultralytics.yolo.utils import LOGGER

    file_handler = logging.FileHandler(log_file_path.as_posix())
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


# def get_pretrained_model_path(dir_path_dict, pretrained_model_name):
#     pretrained_model_path = dir_path_dict['code'] / 'ultralytics/models/v8' / pretrained_model_name
#     return pretrained_model_path


def prepare_train(dir_path_dict, train_kwargs):
    yolo_setting = {
        # No use.
        'datasets_dir': dir_path_dict['data'].as_posix(),  # default datasets directory.
        # Store downloaded models. No use.
        'weights_dir': (dir_path_dict['code'] / 'weights').as_posix(),  # default weights directory.
        # No use.
        'runs_dir': (dir_path_dict['code'] / 'runs').as_posix(),  # default runs directory.
        'uuid': hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # SHA-256 anonymized UUID hash
        'sync': True,  # sync analytics to help with YOLO development
        'api_key': '',  # Ultralytics HUB API key (https://hub.ultralytics.com/)
        'settings_version': '0.0.3'  # Ultralytics settings version
    }

    yolo_config_dir = dir_path_dict['code'] / 'config'

    # Write to setting.yaml
    settings_yaml_file_path = yolo_config_dir / 'settings.yaml'

    settings_yaml_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(settings_yaml_file_path.as_posix(), 'w', encoding='utf-8') as file_stream:
        yaml.safe_dump(yolo_setting, file_stream, sort_keys=False)

    os.environ['YOLO_CONFIG_DIR'] = yolo_config_dir.as_posix()

    # Copy font file to display class names in the screen.
    font_file_path = dir_path_dict['code'] / 'Arial.ttf'
    font_dest_file_path = yolo_config_dir / font_file_path.name

    if not font_dest_file_path.exists():
        assert font_file_path.exists(), font_file_path

        font_dest_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(font_file_path.as_posix(),
                        font_dest_file_path.as_posix())

    # pretrained_model_path
    pretrained_model_name = train_kwargs['pretrained_model_name']
    # pretrained_model_path = get_pretrained_model_path(dir_path_dict, pretrained_model_name)
    pretrained_model_path = Path(yolo_setting['weights_dir']) / pretrained_model_name
    assert pretrained_model_path.exists(), pretrained_model_path


def launch_train(dir_path_dict, train_kwargs):
    from ultralytics import YOLO
    from ultralytics.yolo.engine.model import TASK_MAP
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.yolo.utils import LOGGER, yaml_load, RANK, yaml_save
    from ultralytics.yolo.utils.checks import check_yaml, print_args
    from ultralytics.yolo.utils.files import increment_path

    # Overwrite method train(self, **kwargs)
    class CusomedYOLO(YOLO):
        def train(self, **kwargs):
            """
            Trains the model on a given dataset.

            Args:
                **kwargs (Any): Any number of arguments representing the training configuration.
            """
            self._check_is_pytorch_model()
            if self.session:  # Ultralytics HUB session
                if any(kwargs):
                    LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
                kwargs = self.session.train_args

            ###################################################################
            # Customed: Comment out pip update.
            # -----------------------------------------------------------------
            # check_pip_update_available()
            ###################################################################

            overrides = self.overrides.copy()
            overrides.update(kwargs)
            if kwargs.get('cfg'):
                LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
                overrides = yaml_load(check_yaml(kwargs['cfg']))
            overrides['mode'] = 'train'
            if not overrides.get('data'):
                raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
            if overrides.get('resume'):
                overrides['resume'] = self.ckpt_path
            self.task = overrides.get('task') or self.task
            self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
            if not overrides.get('resume'):  # manually set model only if not resuming
                self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
                self.model = self.trainer.model
            self.trainer.hub_session = self.session  # attach optional HUB session

            ###################################################################
            # Customed: Update trainer.wdir
            # -----------------------------------------------------------------
            # Tensorboard log dir.
            # self.trainer.save_dir = Path(
            #                              increment_path(Path(dir_path_dict['tensorboard']),
            #                                             exist_ok=self.trainer.args.exist_ok if RANK in (-1, 0) else True))

            # Trained weight save dir.
            self.trainer.wdir = Path(dir_path_dict['trained_model'])

            if RANK in (-1, 0):
                self.trainer.wdir.mkdir(parents=True, exist_ok=True)  # make dir
                # self.trainer.args.save_dir = str(self.trainer.save_dir)
                # yaml_save(self.trainer.save_dir / 'args.yaml', vars(self.trainer.args))  # save run args
            self.trainer.last, self.trainer.best = self.trainer.wdir / 'last.pt', self.trainer.wdir / 'best.pt'  # checkpoint paths

            if RANK == -1:
                LOGGER.info('=' * 80)
                reset_format = 'Reset trainer.wdir: {}'
                LOGGER.info(reset_format.format(self.trainer.wdir.as_posix()))
                LOGGER.info('-' * 80)
                print_args(vars(self.trainer.args))
                LOGGER.info('=' * 80)
            ###################################################################

            self.trainer.train()
            # Update model and cfg after training
            if RANK in (-1, 0):
                self.model, _ = attempt_load_one_weight(str(self.trainer.best))
                self.overrides = self.model.args
                self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP

    # log dir.
    add_log_file_handler(dir_path_dict['log'])

    # pretrained_model_path
    pretrained_model_name = train_kwargs['pretrained_model_name']
    # pretrained_model_path = get_pretrained_model_path(dir_path_dict, pretrained_model_name)

    del train_kwargs['pretrained_model_name']

    # Load a pretrained YOLO model (recommended for training)
    model = CusomedYOLO(pretrained_model_name)

    dataset_yaml_file_path = train_kwargs['dataset_yaml_file_path']
    del train_kwargs['dataset_yaml_file_path']

    # Train the model
    results = model.train(data=dataset_yaml_file_path.as_posix(), **train_kwargs)

    LOGGER.info('Completed train.')

    # # Evaluate the model's performance on the validation set
    # results = model.val()



def main():
    project_name = r'illegal_construction'
    environment_name = 'cvmart'

    # 1. For 'yolov8n.pt, download model to weights_dir from
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    # 国内使用
    # !wget https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

    # 2. Download font file to dir_path_dict['code'] from https://ultralytics.com/assets/Arial.ttf

    train_kwargs = dict(
                      pretrained_model_name='yolov8n.pt',
                      epochs=200)

    dir_path_dict = dir_path_dict_collection[environment_name]

    # Comment out because:
    # SyntaxError: 'save_dir' is not a valid YOLO argument. Similar arguments are i.e. ['save_crop=False', 'save=True', 'save_period=-1'].
    # # Tensorboard log dir.
    # train_kwargs['save_dir'] = dir_path_dict['tensorboard']

    # Tensorboard log dir.
    train_kwargs['project'] = dir_path_dict['tensorboard'] / project_name

    # SyntaxError: 'wdir' is not a valid YOLO argument. Similar arguments are i.e. ['save_crop=False', 'save=True', 'save_period=-1'].
    # # Trained weights save dir.
    # train_kwargs['wdir'] = dir_path_dict['trained_model']

    dataset_yaml_file_path = generate_yolo_dataset(project_name, dir_path_dict['data'])
    train_kwargs['dataset_yaml_file_path'] = dataset_yaml_file_path

    if environment_name in ['cvmart', 'colab']:
        prepare_train(dir_path_dict, train_kwargs)
        launch_train(dir_path_dict, train_kwargs)

if __name__ == '__main__':
    main()
