from ultralytics import YOLO

# pip install onnxruntime-gpu


def export_model():
    model = YOLO("/project/train/models/best.pt")
    # model = YOLO("/project/train/src_repo/ultralytics/weights/yolov8n.pt")
    model.export(format='engine', half=True, device=0)


def main():
    print('Start export_model...')
    export_model()
    print('Completed!')


if __name__ == '__main__':
    main()
