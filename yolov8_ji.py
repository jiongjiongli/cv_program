import json
from pathlib import Path
import shutil
from time import time
import logging
import sys
sys.path.append('/project/train/src_repo/ultralytics')

import orjson
import cv2
import torch

from ultralytics import YOLO
from ultralytics.yolo.utils import callbacks, LOGGER
from ultralytics.nn.autobackend import check_class_names
# from export import export_model
from trt_export import export_model
from trt_predict import Predictor


model_file_path = r'/project/train/models/best.pt'
# model_file_path = r'/project/train/models/detect/train/weights/best.pt'
# model_file_path = r'/project/train/src_repo/ultralytics/weights/yolov8n.pt'

log_file_path = r'/project/train/log/log.txt'
draw_image_dir_path = Path('/project/train/src_repo/data/draw')

fake_result = {
    "algorithm_data": {
       "is_alert": False,
       "target_count": 0,
       "target_info": []
    },
    "model_data": {"objects": []}
}

def reset_fake_result():
    fake_result['algorithm_data']['is_alert'] = False
    fake_result['algorithm_data']['target_count'] = 0
    fake_result['algorithm_data']['target_info'] = []
    fake_result['model_data']['objects'] = []


def add_log_file_handler(log_file_path):
    file_handler = logging.FileHandler(Path(log_file_path).as_posix())
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def init():
    '''Initialize model
    Returns: model
    '''
    add_log_file_handler(log_file_path)
    shutil.rmtree(draw_image_dir_path.as_posix())
    draw_image_dir_path.mkdir()

    model = YOLO(model_file_path)

    # Warmup
    input_image_paths = ['/project/train/src_repo/ultralytics/ultralytics/assets/bus.jpg']

    for input_image_path in input_image_paths:
        input_image = cv2.imread(input_image_path)
        # model(input_image, half=True, verbose=False)
        model(input_image, half=True)

    # model.predictor.callbacks = callbacks.get_default_callbacks()
    return model


def process_image(handle=None, input_image=None, args=None, **kwargs):
    start_time = time()
    model = handle
    class_names = getattr(model, 'class_names', model.names)

    results = model(input_image, half=True)

    reset_fake_result()
    pred_object_count = 0

    for result in results:
        # Boxes object for bbox outputs
        boxes = result.boxes

        bboxes = boxes.xyxy
        bboxes_x1y1 = boxes.xyxy[..., :2]
        bboxes_x2y2 = boxes.xyxy[...,2:]
        bboxes_wh = bboxes_x2y2 - bboxes_x1y1

        bboxes_x1y1 = bboxes_x1y1.round().int().tolist()
        bboxes_wh = bboxes_wh.round().int().tolist()
        scores = boxes.conf.float().tolist()
        class_ids = boxes.cls.int().tolist()

        for x1y1, wh, score, class_id in zip(bboxes_x1y1, bboxes_wh, scores, class_ids):
            x1, y1 = x1y1
            w, h = wh
            confidence = score
            class_name = class_names[class_id]
            pred_obj = {
                            "x": x1,
                            "y":y1,
                            "width": w,
                            "height": h,
                            "confidence": confidence,
                            "name": class_name
                       }

            fake_result["model_data"]['objects'].append(pred_obj)

            fake_result["algorithm_data"]["target_info"].append(pred_obj)

            pred_object_count += 1


    if pred_object_count > 0:
        fake_result ["algorithm_data"]["is_alert"] = True
        fake_result ["algorithm_data"]["target_count"] = pred_object_count

    result = orjson.dumps(fake_result).decode()
    end_time = time()

    return result


def main():
    print('Start init...')
    model = init()

    print('Start model predict...')

    image_dir_path = r'/home/data'
    image_dir_path = Path(image_dir_path)
    image_file_paths = image_dir_path.rglob('*.jpg')
    input_image_paths = [image_file_path.as_posix() for image_file_path in image_file_paths]

    # # Warmup
    # model(cv2.imread(input_image_paths[0]), half=True, verbose=False, stream=False)
    durations = []

    for image_index, input_image_path in enumerate(input_image_paths):
        input_image = cv2.imread(input_image_path)

        start_time = time()
        result_str = process_image(handle=model, input_image=input_image)
        end_time = time()

        durations.append(end_time - start_time)

        result = json.loads(result_str)

        if result['algorithm_data']['is_alert']:
            print('input_image_path:', input_image_path)
            print('result:', result)

            for det_obj in result["model_data"]['objects']:
                x = det_obj['x']
                y = det_obj['y']
                obj_width = det_obj['width']
                obj_height = det_obj['height']

                # Start coordinate, here (5, 5)
                # represents the top left corner of rectangle
                start_point = (x, y)

                # Ending coordinate, here (220, 220)
                # represents the bottom right corner of rectangle
                end_point = (x + obj_width, y + obj_height)

                # Blue color in BGR
                color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 2

                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 2 px
                draw_image = cv2.rectangle(input_image, start_point, end_point, color, thickness)

                draw_image_file_path = draw_image_dir_path / Path(input_image_path).name
                cv2.imwrite(draw_image_file_path.as_posix(), draw_image)

    average_duration = sum(durations) / len(durations) * 1000
    print('num_images: {}'.format(len(durations)),
          r'Average duration (ms): {:.1f}'.format(average_duration))
    # print('auto is True for .pt:', model.predictor.dataset.auto)

    # class_names = model.predictor.model.names
    # print('class_names:', class_names)
    # print(results[0])

    print('Completed!')


if __name__ == '__main__':
    main()
