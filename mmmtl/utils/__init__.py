# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .device import auto_select_device
from .distribution import wrap_distributed_model, wrap_non_distributed_model
from .logger import get_root_logger, load_json_log
from .setup_env import setup_multi_processes
from .misc import find_latest_checkpoint
from .tasks import CLASSIFICATION, DETECTION, SEGMENTATION

cls = [
    'collect_env', 'get_root_logger', 'load_json_log', 'setup_multi_processes',
    'wrap_non_distributed_model', 'wrap_distributed_model',
    'auto_select_device', 'find_latest_checkpoint'
]


# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .compat_config import compat_cfg
from .logger import get_caller_name, get_root_logger, log_img_scale
from .memory import AvoidCUDAOOM, AvoidOOM
from .misc import find_latest_checkpoint, update_data_root
from .replace_cfg_vals import replace_cfg_vals
from .rfnext import rfnext_init_model
from .setup_env import setup_multi_processes
from .split_batch import split_batch
from .util_distribution import build_ddp, build_dp, get_device

det = [
    'get_root_logger', 'collect_env', 'find_latest_checkpoint',
    'update_data_root', 'setup_multi_processes', 'get_caller_name',
    'log_img_scale', 'compat_cfg', 'split_batch', 'build_ddp', 'build_dp',
    'get_device', 'replace_cfg_vals', 'AvoidOOM', 'AvoidCUDAOOM',
    'rfnext_init_model','SEGMENTATION','DETECTION','CLASSIFICATION'
]


# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .logger import get_root_logger
from .misc import find_latest_checkpoint
from .set_env import setup_multi_processes
from .util_distribution import build_ddp, build_dp, get_device

seg = [
    'get_root_logger', 'collect_env', 'find_latest_checkpoint',
    'setup_multi_processes', 'build_ddp', 'build_dp', 'get_device'
]

__all__ = list(set(cls) | set(det) | set(seg))


