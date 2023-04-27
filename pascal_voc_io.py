from pathlib import Path
import xml.etree.ElementTree as ET


XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocReader:
    '''From labelImg/libs/pascal_voc_io.py PascalVocReader
    '''
    def parse_xml(self, file_path):
        file_path = Path(file_path).resolve()
        assert file_path.suffix == XML_EXT, 'Unsupported file format'

        xml_tree = ET.parse(file_path.as_posix())
        root = xml_tree.getroot()
        filename = root.find('filename').text

        size = root.find('size')
        size_dict = self.parse_size(size)
        box_list = []

        for object_iter in root.findall('object'):
            bnd_box = object_iter.find('bndbox')
            name = object_iter.find('name').text

            difficult = False

            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))

            bnd_box_dict = self.parse_bnd_box(name, bnd_box, difficult)
            box_list.append(bnd_box_dict)

        parse_result = {'file_name': filename,
                        'size': size_dict,
                        'box_list': box_list}

        return parse_result

    def parse_size(self, size):
        height = int(size.find('height').text)
        width = int(size.find('width').text)
        depth = int(size.find('depth').text)

        size_dict = {'height': height, 'width': width, 'depth': depth}
        return size_dict

    def parse_bnd_box(self, name, bnd_box, difficult):
        x_min = float(bnd_box.find('xmin').text)
        y_min = float(bnd_box.find('ymin').text)
        x_max = float(bnd_box.find('xmax').text)
        y_max = float(bnd_box.find('ymax').text)

        bnd_box_dict = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box_dict['name'] = name
        bnd_box_dict['difficult'] = difficult

        return bnd_box_dict
