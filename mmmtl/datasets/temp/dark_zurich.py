# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .cityscapes import CityscapesDatasetSeg


@DATASETS.register_module()
class DarkZurichDataset(CityscapesDatasetSeg):
    """DarkZurichDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
