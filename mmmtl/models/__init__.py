# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, MTLEARNERS, HEADS, LOSSES, NECKS, ROI_EXTRACTORS, SHARED_HEADS,
                       build_backbone,build_mtlearner, build_head, build_loss, build_neck,
                      build_roi_extractor, build_shared_head)
from .dense_heads import *  # noqa: F401,F403
from .mtlearners import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .plugins import *  # noqa: F401,F403
from .roi_heads import *  # noqa: F401,F403
from .seg_heads import *  # noqa: F401,F403

det = [
    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
    'MTLEARNERS','build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss','build_mtlearner'
]



# # Copyright (c) OpenMMLab. All rights reserved.
# from .backbones import *  # noqa: F401,F403
# from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, build_backbone,
#                       build_head, build_loss, build_segmentor)
# from .decode_heads import *  # noqa: F401,F403
# from .losses import *  # noqa: F401,F403
# from .necks import *  # noqa: F401,F403
# from .segmentors import *  # noqa: F401,F403

# seg = [
#     'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'build_backbone',
#     'build_head', 'build_loss', 'build_segmentor'
# ]



# __all__ = list(set(cls) | set(det) | set(seg))
