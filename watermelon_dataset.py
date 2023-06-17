# 同济子豪兄 2023-2-15
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class WatermelonDataset(BaseSegDataset):
    # 类别和对应的可视化配色
    METAINFO = {
        'classes':['_background_', 'red', 'green', 'white', 'seed-black', 'seed-white'],
        # 'palette':[[132,41,246], [228,193,110], [152,16,60], [58,221,254], [41,169,226], [155,155,155]]
        'palette': [[192,192,192], [255,69,0], [0,255,127], [135,206,250], [0,0,0], [230,230,250]]
    }

    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

