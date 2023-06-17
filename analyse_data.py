from pathlib import Path
import logging
from tqdm import tqdm
import random
import yaml
import shutil
import os
import numpy as np
import pandas as pd
import cv2
from IPython.display import Image, display
from PIL import Image
import matplotlib.pyplot as plt

# The folliwing line is useful in Jupyter notebook
%matplotlib inline

from yolo_dir_path_manager import dir_path_dict_collection
from convert_voc_to_yolo import generate_yolo_dataset
from pascal_voc_io import PascalVocReader


def preprocess(img, target_size):
    width, height = img.shape[:2][::-1]
    target_width, target_height = target_size[:2]

    width_ratio = target_width / width
    height_ratio = target_height / height

    if width_ratio > height_ratio:
        ratio = height_ratio
        unpad_width = min(int(round(width * ratio)), target_width)
        unpad_size = (unpad_width, target_height)
    else:
        ratio = width_ratio
        unpad_height = min(int(round(height * ratio)), target_height)
        unpad_size = (target_width, unpad_height)

    unpad_img = cv2.resize(img, unpad_size, interpolation=cv2.INTER_LINEAR)

    unpad_height, unpad_width = unpad_img.shape[:2]
    left = (target_width - unpad_width) // 2
    top = (target_height - unpad_height) // 2
    right = target_width - unpad_width - left
    bottom = target_height - unpad_height - top
     # add border
    target_img = cv2.copyMakeBorder(unpad_img,
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    cv2.BORDER_CONSTANT,
                                    value=(114, 114, 114))

    return target_img, ratio, left, top


def draw_text(img, text, org, color):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # fontScale
    fontScale = 1

    # org: bottom-left corner of the text string in the image.
    # org = (50, 50)

    # Blue color in BGR
    # color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    img = cv2.putText(img, text, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)

    return img

def draw_image_labels(image_path, box_list, class_index_dict):
    # print(image_path)
    img = cv2.imread(image_path.as_posix())
    target_size = (640, 640)

    colors = ((255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (60, 179, 113),
              (0, 0, 0),
              # (255, 165, 0),
              # (60, 60, 60),
              # (92, 99, 71),
              (0, 255, 255),
              (255, 255, 0),
              (255, 0, 255),
              (106, 90, 205),
              (238, 130, 238))

    img_with_labels, ratio, left, top = preprocess(img, target_size)

    boxes = []
    class_index_list = []

    for box in box_list:
        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']

        class_name = box['name']

        boxes.append([xmin, ymin, xmax, ymax])

        class_index = class_index_dict[class_name]
        class_index_list.append(class_index)

    boxes = np.array(boxes)
    boxes = boxes * ratio + np.array([left, top] * 2)
    boxes = boxes.astype(np.int32)
    text_args_list = []

    for box, class_index in zip(boxes, class_index_list):
        # if class_index in [1]:
        #     continue

        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = tuple(box[:2])

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = tuple(box[2:])

        # Blue color in BGR
        color = colors[class_index]

        # Line thickness of 2 px
        thickness = 2

        # print(start_point, end_point, img_with_labels.shape)

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img_with_labels = cv2.rectangle(img_with_labels, start_point, end_point, color, thickness)

        text_args_list.append((str(class_index), end_point, color))

    for text_args in text_args_list:
        img_with_labels = draw_text(img_with_labels, *text_args)

    img_with_labels = cv2.cvtColor(img_with_labels, cv2.COLOR_BGR2RGB)
    img_with_labels = Image.fromarray(img_with_labels, 'RGB')

    # print('display...')
    # Display the Numpy array as Image
    display(img_with_labels)
    # plt.imshow(img_with_labels)
    # plt.show()


def analyze(dataset_yaml_file_path):
    with open(Path(dataset_yaml_file_path).as_posix(), 'r', encoding='utf-8') as file_stream:
        dataset = yaml.safe_load(file_stream)

    root_path = dataset['path']
    root_path = Path(root_path)

    train_index_path = root_path / dataset['train']
    val_index_path = root_path / dataset['val']
    class_name_dict = {int(class_index): class_name for class_index, class_name in dataset['names'].items()}
    class_index_dict = {class_name: class_index for class_index, class_name in class_name_dict.items()}

    with open(train_index_path.as_posix(), 'r', encoding='utf-8') as file_stream:
        train_image_file_paths = [line.strip() for line in file_stream]

    with open(val_index_path.as_posix(), 'r', encoding='utf-8') as file_stream:
        val_image_file_paths = [line.strip() for line in file_stream]

    image_sizes = []
    gt_wh_dict = {}
    object_count_infos = []
    image_file_paths = train_image_file_paths + val_image_file_paths

    for image_file_path in image_file_paths:
        image_file_path = image_file_path.strip()

        if not image_file_path:
            continue

        image_file_path = Path(image_file_path)

        image_height, image_width = cv2.imread(image_file_path.as_posix()).shape[:2]
        image_sizes.append([image_width, image_height])

        label_file_path = image_file_path.with_suffix('.txt')

        with open(label_file_path, 'r', encoding='utf-8') as file_stream:
            labeled_objects = [line.strip() for line in file_stream]

        image_object_count_dict = {}

        for labeled_object in labeled_objects:
            if not labeled_object:
                continue

            class_index, x_center, y_center, w, h = labeled_object.split()
            class_index = int(class_index)
            x_center, y_center, w, h = [float(item) for item in [x_center, y_center, w, h]]

            gt_wh_dict.setdefault(class_index, [])
            gt_wh_dict[class_index].append([w, h])

            image_object_count_dict.setdefault(class_index, 0)
            image_object_count_dict[class_index] +=1

        object_count_infos.append(image_object_count_dict)

    image_sizes = pd.DataFrame(image_sizes, columns=['width', 'height'])
    image_sizes['ratio'] = image_sizes['width'] / image_sizes['height']
    print('image_sizes')
    print(image_sizes.describe())

    gt_whs_dict = {}
    object_counts_dict = {}

    for class_index, class_name in class_name_dict.items():
        gt_whs = gt_wh_dict.get(class_index, [])
        gt_whs = pd.DataFrame(gt_whs, columns=['width', 'height'])

        gt_whs_dict[class_name] = gt_whs * 640

        object_counts = [object_count_info.get(class_index, 0)
                         for object_count_info in object_count_infos]

        object_counts_dict[class_name] = object_counts

    pd.options.display.max_columns = len(object_counts_dict)

    for class_name, gt_whs in gt_whs_dict.items():
        print(class_name)
        print(gt_whs.describe())

    obj_counts = pd.DataFrame(object_counts_dict)

    print('object_counts')
    print(obj_counts.describe())

    print(obj_counts.idxmax())
    print(obj_counts.max())

    image_paths = [image_file_paths[file_index] for class_index, file_index in obj_counts.idxmax().iteritems()]

    reader = PascalVocReader()

    for image_path in image_paths:
        image_path = Path(image_path)
        voc_file_path = image_path.with_suffix('.xml')
        parse_result = reader.parse_xml(voc_file_path)
        size_dict = parse_result['size']
        img_shape = (size_dict['height'], size_dict['width'], size_dict['depth'])
        box_list = parse_result['box_list']

        draw_image_labels(image_path, box_list, class_index_dict)

def main():
    project_name = r'illegal_construction'
    environment_name = 'local'

    dir_path_dict = dir_path_dict_collection[environment_name]

    dataset_yaml_file_path = generate_yolo_dataset(project_name, dir_path_dict['data'])
    analyze(dataset_yaml_file_path)


if __name__ == '__main__':
    main()
