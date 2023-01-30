# Copyright (c) OpenMMLab. All rights reserved.
from .builder import SEG_DATASETS
from .cityscapes import CityscapesDataset


@SEG_DATASETS.register_module()
class DarkZurichDataset(CityscapesDataset):
    """DarkZurichDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
