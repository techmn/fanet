# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class ZeroWasteDataset(CustomDataset):
    """ZeroWaste dataset.
    """
    METAINFO = dict(
        classes=('background','rigid_plastic', 'cardboard', 'metal', 'soft_plastic'),
        palette=[[10, 10, 10], [6, 230, 230],[4, 200, 3], [204, 5, 255], [235, 255, 7]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
