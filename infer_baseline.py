import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
from time import time

# sys.path.append('/project/train/src_repo/ultralytics')

from ultralytics import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG


class BaseInferManager:
    def __init__(self, model=None, imgsz=None):
        self.model = model
        self.imgsz = imgsz if imgsz is not None else DEFAULT_CFG.imgsz
        self.device = self.model.device if self.model else torch.device('cuda:0')
        self.start_time = time()

    def add_letterbox(self, new_shape, img):
        """Return updated labels and image with added border."""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        return img

    def _single_preprocess(self, im):
        # train/src_repo/ultralytics/ultralytics/yolo/engine/predictor.py
        # train/src_repo/ultralytics/ultralytics/yolo/data/dataloaders/stream_loaders.py
        """Preprocesses a single image for inference."""

        # Jiongjiong
        from time import time
        start_time = time()

        im = self.add_letterbox(self.imgsz, im)

        end_time = time()
        print('LetterBox durating:', r'{:.2f}ms'.format((end_time - start_time) * 1000))

        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        end_time = time()
        print('_single_preprocess durating:', r'{:.2f}ms'.format((end_time - start_time) * 1000))
        return im

    def preprocess(self, im):
        im = self._single_preprocess(im)
        im =  im[None]

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im


    def postprocess(self, preds, img, orig_img):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = self.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):

            if not isinstance(orig_img, torch.Tensor):
                pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            orig_shape = orig_img.shape[:2]
            boxes = pred

            if boxes.ndim == 1:
                boxes = boxes[None, :]

            xyxy = boxes[:, :4]
            conf = boxes[:, -2]
            cls = boxes[:, -1]
            self.model.names

            results.append(boxes)
        return results

    def clip_boxes(self, boxes, shape):
        """
        It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
        shape

        Args:
          boxes (torch.Tensor): the bounding boxes to clip
          shape (tuple): the shape of the image
        """
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """
        Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
        (img1_shape) to the shape of a different image (img0_shape).

        Args:
          img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
          boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
          img0_shape (tuple): the shape of the target image, in the format of (height, width).
          ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                             calculated based on the size difference between the two images.

        Returns:
          boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """

        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes



    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            nc=0,  # number of classes (optional)
                            max_time_img=0.05,
                            max_nms=30000,
                            max_wh=7680,
                            ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose(0, -1)[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)
            box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            if multi_label:
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output

    def print_duration(self, comment, start_time=None, end_time=None, duration_sec=None):
        if not duration_sec:
            if not start_time:
                start_time = self.start_time

            if not end_time:
                end_time = time()

            duration_sec = end_time - start_time

        print('{} duration: {:.2f}ms'.format(comment, duration_sec * 1000))


    def infer(self, img):
        # self.start_time = time()
        im = self.add_letterbox(self.imgsz, img)
        # self.print_duration('add_letterbox')

        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im =  im[None]

        im = torch.from_numpy(im).to(self.device)
        im = im.half()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # print(im.shape)

        # self.print_duration('preprocess')
        # return im

        preds = self.model(im)
        # results = self.postprocess(preds, im, img)


def main():
    # model = YOLO('/project/train/src_repo/ultralytics/model_backup/best.engine')

    model = YOLO("/project/train/src_repo/ultralytics/weights/yolov8n.pt")

    from ultralytics.nn.modules import C2f,  Detect

    det_modules = model.model.model

    for det_module in det_modules:
        if isinstance(det_module, C2f):
            det_module.forward = det_module.forward_split = det_module.forward_no_split

        if isinstance(det_module, Detect):
            det_module.forward = det_module.forward_nms


    image_dir_path = r'/home/data'
    image_dir_path = Path(image_dir_path)
    image_file_paths = image_dir_path.rglob('*.jpg')
    input_image_paths = [image_file_path.as_posix() for image_file_path in image_file_paths]

    # Warmup
    # model(cv2.imread(input_image_paths[0]), half=True, verbose=False, stream=False)

    # manager = BaseInferManager(model.predictor.model, model.predictor.model.imgsz)
    manager = BaseInferManager()
    durations = []

    for file_index, input_image_path in enumerate(input_image_paths):
        input_image = cv2.imread(input_image_path)

        start_time = time()
        results = manager.infer(input_image)
        end_time = time()

        if file_index > 0:
            durations.append(end_time - start_time)

        assert results.shape == torch.Size([1, 3, 640, 640]), results.shape

    average_duration_sec = sum(durations) / len(durations)
    manager.print_duration('Images: {}, Average'.format(len(durations)), duration_sec=average_duration_sec)


if __name__ == '__main__':
    main()
