# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry
from mmmtl.utils.tasks import *
from .det import build_dataset_det,build_dataloader_det
from .seg import build_dataset_seg,build_dataloader_seg
from .cls import build_dataset_cls,build_dataloader_cls

# import mmmtl.datasets as detdataset
# import mmmtl.datasets as segdataset
# DET_DATASETS = detdataset.DATASETS
# SEG_DATASETS = segdataset.DATASETS

def build_dataset(cfg, default_args=None,task=DETECTION):
    if task==DETECTION:
        return build_dataset_det(cfg,default_args)
    elif task==SEGMENTATION:
        return build_dataset_seg(cfg,default_args)
    return



def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     round_up=True,
                     seed=None,
                     runner_type='IterBasedRunner',
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=True,
                     sampler_cfg=None,
                     class_aware_sampler=None,
                     task=DETECTION,
                     **kwargs
                     ):
    
    if task==DETECTION:
        return build_dataloader_det(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus,
                     dist,
                     shuffle,
                     seed,
                     runner_type,
                     persistent_workers,
                     class_aware_sampler,
                     **kwargs)
    if task==SEGMENTATION:
        return build_dataloader_seg(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus,
                     dist,
                     shuffle,
                     seed,
                     drop_last,
                     pin_memory,
                     persistent_workers,
                     **kwargs)
    return


