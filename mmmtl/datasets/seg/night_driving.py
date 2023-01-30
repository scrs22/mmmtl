# Copyright (c) OpenMMLab. All rights reserved.
from .builder import SEG_DATASETS
from .cityscapes import CityscapesDataset


@SEG_DATASETS.register_module()
class NightDrivingDataset(CityscapesDataset):
    """NightDrivingDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtCoarse_labelTrainIds.png',
            **kwargs)
