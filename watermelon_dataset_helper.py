import io
from pathlib import Path
import shutil

from PIL import Image, ImageOps
import numpy as np
import cv2


def verify_ann_file(file_path, num_classes):
    with open(Path(file_path).as_posix(), 'rb') as f:
        img_bytes = f.read()

    flag='unchanged'
    channel_order = 'bgr'

    with io.BytesIO(img_bytes) as buff:
        img = Image.open(buff)
        # img = _pillow2array(img, flag, channel_order)
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    # print(np.unique(array))
    assert array.ndim == 2

    img_arr = cv2.imread(Path(file_path).as_posix(), cv2.IMREAD_UNCHANGED)
    assert np.all(array == img_arr), file_path

    assert np.all((0 <= array) & (array <= num_classes - 1)), np.unique(array)

    class_index_arr, per_class_label_counts = np.unique(array, return_counts=True)
    return class_index_arr, per_class_label_counts

def verify_ann_files(ann_root_path, num_classes):
    file_paths = list(Path(ann_root_path).rglob('*.*'))

    print('file_paths count:', len(file_paths))
    label_counts = np.zeros(num_classes, dtype=np.int64)

    for file_path in file_paths:
        # file_path = r'data/watermelon87_database/masks/00026.png'
        class_index_arr, per_class_label_counts = verify_ann_file(file_path, num_classes)

        label_counts[class_index_arr] += per_class_label_counts

    # label_counts = label_counts / np.sum(label_counts, dtype=label_counts.dtype)
    # print('label_counts:', label_counts, label_counts.dtype)
    weights = np.sum(label_counts, dtype=label_counts.dtype) / label_counts
    weights = weights / np.sum(weights)
    print('weights:', weights, weights.dtype)

def find_file_mapping(data_root_path):
    data_root_path = Path(data_root_path)

    image_dir_path = data_root_path / 'img_dir'
    ann_dir_path = data_root_path / 'ann_dir'

    image_file_paths = list(image_dir_path.rglob('*.jpg'))
    ann_file_paths = list(ann_dir_path.rglob('*.png'))
    file_path_mappings = []

    for image_file_path in image_file_paths:
        image_file_name = image_file_path.stem

        matched_ann_file_paths = []

        for ann_file_path in ann_file_paths:
            ann_file_name = ann_file_path.stem

            if image_file_name == ann_file_name:
                matched_ann_file_paths.append(ann_file_path)


        if len(matched_ann_file_paths) == 1:
            file_path_mapping = {
                              'image_file_path': image_file_path,
                              'ann_file_path': matched_ann_file_paths[0]
                             }
            file_path_mappings.append(file_path_mapping)
            continue

        assert len(matched_ann_file_paths) == 0, matched_ann_file_paths

        for ann_file_path in ann_file_paths:
            ann_file_name = ann_file_path.stem

            if image_file_name.startswith(ann_file_name):
                matched_ann_file_paths.append(ann_file_path)

        if len(matched_ann_file_paths) == 1:
            file_path_mapping = {
                              'image_file_path': image_file_path,
                              'ann_file_path': matched_ann_file_paths[0]
                             }
            print(file_path_mapping)
            file_path_mappings.append(file_path_mapping)
            continue

        assert len(matched_ann_file_paths) == 0, matched_ann_file_paths
        print(r'Erro! No matched_ann_file_paths for : {}'.format(image_file_path))

    return file_path_mappings


def generate_database(data_root_path, database_root_path):
    data_root_path = Path(data_root_path)
    database_root_path = Path(database_root_path)

    file_path_mappings = find_file_mapping(data_root_path)

    output_file_mappings = []

    for file_index, file_path_mapping in enumerate(file_path_mappings):
        image_file_path = file_path_mapping['image_file_path']
        ann_file_path = file_path_mapping['ann_file_path']

        assert image_file_path.suffix in ['.jpg'], image_file_path
        assert ann_file_path.suffix in ['.png'], ann_file_path

        output_image_file_name = r'{:05}.jpg'.format(file_index)
        output_ann_file_name   = r'{:05}.png'.format(file_index)

        output_image_file_path = database_root_path / image_file_path.parent.relative_to(data_root_path) / output_image_file_name
        output_ann_file_path   = database_root_path / ann_file_path.parent.relative_to(data_root_path)   / output_ann_file_name

        output_image_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_file_path.as_posix(), output_image_file_path.as_posix())

        output_ann_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ann_file_path.as_posix(), output_ann_file_path.as_posix())

        output_file_mapping = {
            'image_file_path': image_file_path.as_posix(),
            'output_image_file_path': output_image_file_path.as_posix(),
            'ann_file_path': ann_file_path.as_posix(),
            'output_ann_file_path': output_ann_file_path.as_posix()
        }

        output_file_mappings.append(output_file_mapping)

    file_mappings_path = database_root_path / 'file_mappings.json'

    with open(file_mappings_path.as_posix(), 'w') as file_stream:
        json.dump(output_file_mappings, file_stream, indent=4)



def main():
    num_classes = 6
    # data_root_path = 'data/Watermelon87_Semantic_Seg_Mask'
    # database_root_path = 'data/Watermelon87_Seg_DB'
    # generate_database(data_root_path, database_root_path)
    # print(r'Completed generate_database to {}!'.format(database_root_path))

    database_root_path = r'data/watermelon87_database'
    ann_root_path = Path(database_root_path) / 'ann_dir'
    # ann_root_path = r'data/watermelon87_database/masks'
    verify_ann_files(ann_root_path, num_classes)


if __name__ == '__main__':
    main()
