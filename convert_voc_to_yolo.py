from pascal_voc_io import PascalVocReader, XML_EXT
from yolo_io import YOLOWriter, TXT_EXT
from yolo_dir_path_manager import dir_path_dict_collection

from pathlib import Path
import logging
from tqdm import tqdm
import random
import yaml
import shutil
import os


class Converter:
    def __init__(self):
        self.reader = PascalVocReader()
        self.writer = YOLOWriter()

    def save_as_yolo_format(self, voc_file_path, class_names):
        '''From labelImg/libs/labelFile.py LabelFile.save_yolo_format
        '''
        result = {}

        voc_file_path = Path(voc_file_path).resolve()
        parse_result = self.reader.parse_xml(voc_file_path)
        file_name = parse_result['file_name']

        image_file_path = voc_file_path.parent / Path(file_name).name

        if not image_file_path.exists():
            logging.warn('File not exist! {}'.format(image_file_path))
            return result

        result['image_file_path'] = image_file_path

        size_dict = parse_result['size']
        img_shape = (size_dict['height'], size_dict['width'], size_dict['depth'])

        box_list = parse_result['box_list']
        target_file_path = voc_file_path.with_suffix(TXT_EXT)

        self.writer.save(box_list, class_names, img_shape, target_file_path)

        return result


def generate_data_index_file(image_file_paths,
                              data_index_file_path):
    with open(data_index_file_path.as_posix(), 'w', encoding='utf-8') as file_stream:
        for image_file_path in image_file_paths:
            file_stream.write('{}\n'.format(image_file_path.as_posix()))


def generate_dataset_yaml_file(data_index_file_name_dict,
                               class_name_dict,
                               data_dir_path,
                               dataset_yaml_file_path):
    content = {'path': data_dir_path.as_posix()}

    for mode, file_name in data_index_file_name_dict.items():
        content[mode] = file_name

    content['names'] = class_name_dict

    with open(dataset_yaml_file_path.as_posix(), 'w', encoding='utf-8') as file_stream:
        yaml.safe_dump(content, file_stream, sort_keys=False)


def generate_yolo_dataset(project_name, data_dir_path):
    data_dir_path = Path(data_dir_path).resolve()
    converter = Converter()
    class_names = []
    image_file_paths = []

    voc_file_paths = list(data_dir_path.rglob('*{}'.format(XML_EXT)))
    success_files = 0
    completed_files = 0

    # progress_bar = tqdm(voc_file_paths)

    for voc_file_path in voc_file_paths:
        try:
            save_result = converter.save_as_yolo_format(voc_file_path, class_names)

            if save_result:
                image_file_path = save_result['image_file_path']
                image_file_paths.append(image_file_path)
                success_files += 1
        except:
            logging.exception('Exception occurred during save_as_yolo_format!')
            raise

        completed_files += 1

        # message_format = 'success_files: {} completed_files: {} / {}'
        # message = message_format.format(success_files, completed_files, len(voc_file_paths))
        # progress_bar.set_description(message)

    print('class_names', class_names)

    # Train val split and index file generation.
    seed = 1009
    random.seed(seed)
    random.shuffle(image_file_paths)
    train_percent = 0.9

    train_samples_count = int(len(image_file_paths) * train_percent)
    image_file_path_dict = {
                'train': image_file_paths[:train_samples_count],
                'val': image_file_paths[train_samples_count:]
              }

    data_index_file_name_dict = {}

    for mode in ['train', 'val']:
        data_index_file_name = '{}.txt'.format(mode)
        data_index_file_name_dict[mode] = data_index_file_name

    for mode, data_index_file_name in data_index_file_name_dict.items():
        data_index_file_path = data_dir_path / data_index_file_name
        generate_data_index_file(image_file_path_dict[mode],
                                  data_index_file_path)


    # Generate dataset.yaml file.
    class_name_dict = {class_index: class_name
                       for class_index, class_name in enumerate(class_names)}

    dataset_yaml_file_path = data_dir_path / '{}.yaml'.format(project_name)
    generate_dataset_yaml_file(data_index_file_name_dict,
                               class_name_dict,
                               data_dir_path,
                               dataset_yaml_file_path)

    return dataset_yaml_file_path


def main():
    project_name = r'illegal_construction'
    environment_name = 'local'

    dir_path_dict = dir_path_dict_collection[environment_name]

    dataset_yaml_file_path = generate_yolo_dataset(project_name, dir_path_dict['data'])

if __name__ == '__main__':
    main()
