# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry
from mmmtl.utils.tasks import *
from .det import build_dataset_detection,build_dataloader_detection
from .seg import build_dataset_segmentation,build_dataloader_segmentation
from .cls import build_dataset_classification,build_dataloader_classification

# import mmmtl.datasets as detdataset
# import mmmtl.datasets as segdataset
# DET_DATASETS = detdataset.DATASETS
# SEG_DATASETS = segdataset.DATASETS

def build_dataset(task=DETECTION,*args,**kwargs):
    method=eval(f"build_dataset_{task}")
    return method(*args,**kwargs)




def build_dataloader(task=DETECTION,
                     *args,
                     **kwargs
                     ):
    
    method=eval(f"build_dataloader_{task}")
    return method(*args,**kwargs)