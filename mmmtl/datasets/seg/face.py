# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import SEG_DATASETS
from .custom import CustomDataset


@SEG_DATASETS.register_module()
class FaceOccludedDataset(CustomDataset):
    """Face Occluded dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'face')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, split, **kwargs):
        super(FaceOccludedDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
