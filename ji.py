import json
from pathlib import Path
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


pt_model_path = r'/project/train/models/best.pt'
log_file_path = r'/project/train/log/log.txt'

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
    export_model(pt_model_path)

    # Load a model
    # model = YOLO("/project/train/models/best.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("/project/train/models/best.engine")
    engine_model_path = Path(pt_model_path).with_suffix('.engine').as_posix()
    model = Predictor(engine_model_path)

    model.class_names = check_class_names(model.class_names)

    # Warmup
    input_image_paths = ['/project/train/src_repo/ultralytics/ultralytics/assets/bus.jpg']

    for input_image_path in input_image_paths:
        input_image = cv2.imread(input_image_path)
        # model(input_image, half=True, verbose=False, stream=False)
        model(input_image)

    # model.predictor.callbacks = callbacks.get_default_callbacks()
    return model


def process_image(handle=None, input_image=None, args=None, **kwargs):
    start_time = time()
    '''Do inference to analysis input_image and get output
    Attributes:
    handle: algorithm handle returned by init()
    input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
    Returns: process result
    '''
    # start_time = time()
    # Process image here
    model = handle
    class_names = model.class_names
    # results = model.predictor(source=input_image, stream=False)
    # results = model(input_image, half=True, verbose=False, stream=False)
    results = model(input_image)
    # model_time = time()
    # fake_result = {}
    # fake_result["algorithm_data"] = {
    #     "is_alert": False,
    #     "target_count": 0,
    #     "target_info": []
    # }
    # fake_result["model_data"] = {}
    # fake_result["model_data"]["objects"] = [
    #     {
    #         "x": 100,
    #         "y": 150,
    #         "width": 70,
    #         "height": 400,
    #         "confidence": 0.999660,
    #         "name": "person"
    #     },
    #     {
    #         "x": 100,
    #         "y": 150,
    #         "width": 40,
    #         "height": 20,
    #         "confidence": 0.999660,
    #         "name": "red_hat"
    #     }
    # ]

    #  fake_result = {}

    #  fake_result["algorithm_data"] = {
    #     "is_alert": False,
    #     "target_count": 0,
    #     "target_info": []
    # }
    #  fake_result["model_data"] = {"objects": []}
    reset_fake_result()
    # Process detections
    cnt = 0

    # for result in results:
        # class_names = result.names
        # boxes = result.boxes

        # xyxy_tensor = boxes.xyxy
        # conf_tensor = boxes.conf
        # cls_tensor  = boxes.cls

    # for xyxy, conf, cls in zip(xyxy_tensor, conf_tensor, cls_tensor):
    #     xyxy = [int(coordinate.item()) for coordinate in xyxy]
        # confidence = float(conf.item())

    bboxes, scores, labels = results

    # print('scores:', scores)
    # print('input_image.shape:', input_image.shape)
    # print('bboxes:', bboxes)

    bboxes_x1y1 = bboxes[..., :2]
    bboxes_x2y2 = bboxes[...,2:]
    bboxes_wh = bboxes_x2y2 - bboxes_x1y1

    # assert (bboxes_wh > 0).all(), bboxes_wh

    # bbox_index_arr = torch.nonzero((wh > 0).all(-1)).view(-1)

    filter_result_time = time()

    # if bbox_index_arr.numel() > 0:
    #     for bbox_index in bbox_index_arr:
            # x1, y1 = x1y1[bbox_index].round().int().tolist()
            # w, h = wh[bbox_index].round().int().tolist()
            # confidence = float(scores[bbox_index])
            # cls_id = int(labels[bbox_index])

    for x1y1, wh, score, label in zip(bboxes_x1y1, bboxes_wh, scores, labels):
        x1, y1 = x1y1.round().int().tolist()
        w, h = wh.round().int().tolist()
        confidence = float(score)
        cls_id = int(label)

        class_name = class_names[cls_id]

        fake_result["model_data"]['objects'].append({
                "x": x1,
                "y":y1,
                "width": w,
                "height": h,
                "confidence": confidence,
                "name": class_name
        })
        fake_result["algorithm_data"]["target_info"].append({
            "x": x1,
            "y":y1,
            "width": w,
            "height": h,
            "confidence": confidence,
            "name": class_name
        })

        cnt += 1


    # for i, det in enumerate(pred):  # detections per image
    #     gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #     print(len(det))
    #     if det:
    #         # Rescale boxes from img_size to im0 size
    #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    #         for *xyxy, conf, cls in det:
    #             if conf < prob_thres:
    #                 continue
    #             fake_result["model_data"]['objects'].append({
    #                 "x":int(xyxy[0]),
    #                 "y":int(xyxy[1]),
    #                 "width":int(xyxy[2]-xyxy[0]),
    #                 "height":int(xyxy[3]-xyxy[1]),
    #                 "confidence":float(conf),
    #                 "name":names[int(cls)]
    #             })
    #             cnt+=1
    #             fake_result["algorithm_data"]["target_info"].append({
    #                 "x":int(xyxy[0]),
    #                 "y":int(xyxy[1]),
    #                 "width":int(xyxy[2]-xyxy[0]),
    #                 "height":int(xyxy[3]-xyxy[1]),
    #                 "confidence":float(conf),
    #                 "name":names[int(cls)]
    #             }
    #             )
    if cnt>0:
        fake_result ["algorithm_data"]["is_alert"] = True
        fake_result ["algorithm_data"]["target_count"] = cnt
    else:
        fake_result ["algorithm_data"]["target_info"]=[]

    # before_json_time = time()
    # result = json.dumps(fake_result, indent=4).decode()
    result = orjson.dumps(fake_result).decode()
    end_time = time()

    time_list = [('pre_resize_time', model.pre_resize_time),
                 ('resize_time', model.resize_time),
                 ('letterbox_time', model.letterbox_time),
                 ('preprocess_time', model.preprocess_time),
                 ('pre_execute_time', model.model.pre_execute_time),
                 ('post_execute_time', model.model.post_execute_time),
                 ('post_model_time', model.post_model_time),
                 ('filter_result_time', filter_result_time),
                 ('end_time', end_time)
                ]

    duration_strs = []

    for task_name, task_end_time in time_list:
        duration_ms = (task_end_time - start_time) * 1000
        duration_str = r'[{}: {:.1f}]'.format(task_name, duration_ms)
        duration_strs.append(duration_str)

    LOGGER.info(', '.join(duration_strs))

    # print('model_time:', r'{:.2f}ms'.format((model_time - start_time) * 1000))
    # print('before_json_time:', r'{:.2f}ms'.format((before_json_time - start_time) * 1000))
    # print('Duration:', r'{:.2f}ms'.format((end_time - start_time) * 1000))


    return result


def main():
    # export_model()
    # model = YOLO("/project/train/models/best.engine")
    # model = YOLO('/project/train/src_repo/ultralytics/model_backup/best.engine')
    # model = YOLO("/project/train/src_repo/ultralytics/weights/yolov8n.pt")
    # model = YOLO("/project/train/src_repo/ultralytics/weights/yolov8n.engine")
    # input_image_paths = ['/project/train/src_repo/ultralytics/ultralytics/assets/bus.jpg']

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

                draw_image_file_path = Path('/project/train/src_repo/data/draw') / Path(input_image_path).name
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
