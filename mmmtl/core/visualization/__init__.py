# Copyright (c) OpenMMLab. All rights reserved.
from .image import (BaseFigureContextManager, ImshowInfosContextManager,
                    color_val_matplotlib, imshow_infos)

cls = [
    'BaseFigureContextManager', 'ImshowInfosContextManager', 'imshow_infos',
    'color_val_matplotlib'
]

# Copyright (c) OpenMMLab. All rights reserved.
from .image import (color_val_matplotlib, imshow_det_bboxes,
                    imshow_gt_det_bboxes)
from .palette import get_palette, palette_val

det = [
    'imshow_det_bboxes', 'imshow_gt_det_bboxes', 'color_val_matplotlib',
    'palette_val', 'get_palette'
]


__all__ = list(set(cls) | set(det))
