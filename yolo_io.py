from pathlib import Path


TXT_EXT = '.txt'
ENCODE_METHOD = 'utf-8'


class YOLOWriter:
    '''From labelImg/libs/yolo_io.py YOLOWriter
    '''
    def bnd_box_to_yolo_line(self, box, class_list, img_shape):
        x_min = box['xmin']
        x_max = box['xmax']
        y_min = box['ymin']
        y_max = box['ymax']

        x_center = float((x_min + x_max)) / 2 / img_shape[1]
        y_center = float((y_min + y_max)) / 2 / img_shape[0]

        w = float((x_max - x_min)) / img_shape[1]
        h = float((y_max - y_min)) / img_shape[0]

        box_name = box['name']

        if box_name not in class_list:
            class_list.append(box_name)

        class_index = class_list.index(box_name)

        return class_index, x_center, y_center, w, h

    def save(self, box_list, class_list, img_shape, target_file):
        target_file = Path(target_file).resolve()
        assert target_file.suffix == TXT_EXT, 'Unsupported file format: {}'.format(target_file.suffix)

        with open(target_file.as_posix(), 'w', encoding=ENCODE_METHOD) as out_file:
            for box in box_list:
                class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(box, class_list, img_shape)
                out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, x_center, y_center, w, h))
