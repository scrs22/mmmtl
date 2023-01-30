# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
# from .det import *
# from .seg import *
# from .cls import *
from .builder import  build_dataloader, build_dataset
from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler

__all__=['build_dataloader', 'build_dataset', 'DistributedGroupSampler', 'DistributedSampler', 'GroupSampler']

# seg = [
#     'CustomDatasetSeg', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
#     'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDatasetSeg',
#     'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
#     'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
#     'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
#     'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDatasetSeg','MultiImageMixDatasetDet',
#     'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset', 'FaceOccludedDataset',
#     'ImageNetSDataset', 'LoadImageNetSAnnotations',
#     'LoadImageNetSImageFromFile','SAMPLERS','DistributedGroupSampler', 'DistributedSampler', 'GroupSampler'
# ]

# from .deepfashion import DeepFashionDataset
# from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
# from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
# from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
# from .utils import (NumClassCheckHook, get_loading_pipeline,
#                     replace_ImageToTensor)
# from .voc import VOCDataset
# from .wider_face import WIDERFaceDataset
# from .xml_style import XMLDataset

# det = [
#     'CustomDatasetDet', 'XMLDataset', 'CocoDataset', 'DeepFashionDataset',
#     'VOCDataset', 'CityscapesDatasetDet', 'LVISDataset', 'LVISV05Dataset',
#     'LVISV1Dataset', 'GroupSampler', 'DistributedGroupSampler',
#     'DistributedSampler', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
#     'ClassBalancedDataset', 'WIDERFaceDataset', 'DATASETS', 'PIPELINES',
#     'build_dataset', 'replace_ImageToTensor', 'get_loading_pipeline',
#     'NumClassCheckHook', 'CocoPanopticDataset', 'MultiImageMixDataset',
#     'OpenImagesDataset', 'OpenImagesChallengeDataset'
# ]

# # Copyright (c) OpenMMLab. All rights reserved.
# from .base_dataset import BaseDataset
# from .builder import (DATASETS, PIPELINES, SAMPLERS, build_dataloader,
#                       build_dataset, build_sampler)
# from .cifar import CIFAR10, CIFAR100
# from .cub import CUB
# from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
#                                KFoldDataset, RepeatDataset)
# from .imagenet import ImageNet
# from .imagenet21k import ImageNet21k
# from .mnist import MNIST, FashionMNIST
# from .multi_label import MultiLabelDataset
# from .samplers import DistributedSampler, RepeatAugSampler
# from .stanford_cars import StanfordCars
# from .voc import VOC

# cls = [
#     'BaseDataset', 'ImageNet', 'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST',
#     'VOC', 'MultiLabelDataset', 'build_dataloader', 'build_dataset',
#     'DistributedSampler', 'ConcatDataset', 'RepeatDataset',
#     'ClassBalancedDataset', 'DATASETS', 'PIPELINES', 'ImageNet21k', 'SAMPLERS',
#     'build_sampler', 'RepeatAugSampler', 'KFoldDataset', 'CUB',
#     'CustomDatasetCls', 'StanfordCars'
# ]

# __all__ = list(set(cls) | set(det) | set(seg))