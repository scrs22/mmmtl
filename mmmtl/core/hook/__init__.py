# # Copyright (c) OpenMMLab. All rights reserved.
# from .class_num_check_hook import ClassNumCheckHook
# from .lr_updater import CosineAnnealingCooldownLrUpdaterHook
# from .precise_bn_hook import PreciseBNHook
# from .wandblogger_hook import MMClsWandbHook

# cls = [
#     'ClassNumCheckHook', 'PreciseBNHook',
#     'CosineAnnealingCooldownLrUpdaterHook', 'MMClsWandbHook'
# ]

# # Copyright (c) OpenMMLab. All rights reserved.
# from .wandblogger_hook import mmmtlWandbHook

# seg= ['mmmtlWandbHook']


# Copyright (c) OpenMMLab. All rights reserved.
from .checkloss_hook import CheckInvalidLossHook
from .ema import ExpMomentumEMAHook, LinearMomentumEMAHook
from .memory_profiler_hook import MemoryProfilerHook
from .set_epoch_info_hook import SetEpochInfoHook
from .sync_norm_hook import SyncNormHook
from .sync_random_size_hook import SyncRandomSizeHook
from .wandblogger_hook import MMDetWandbHook,MMClsWandbHook,MMSegWandbHook
from .yolox_lrupdater_hook import YOLOXLrUpdaterHook
from .yolox_mode_switch_hook import YOLOXModeSwitchHook

__all__ = [
    'SyncRandomSizeHook', 'YOLOXModeSwitchHook', 'SyncNormHook',
    'ExpMomentumEMAHook', 'LinearMomentumEMAHook', 'YOLOXLrUpdaterHook',
    'CheckInvalidLossHook', 'SetEpochInfoHook', 'MemoryProfilerHook',
    'MMDetWandbHook','MMSegWandbHook','MMClsWandbHook'
]

