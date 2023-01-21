# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS
MTLEARNERS = MODELS
# DETECTORS = MODELS
# SEGMENTORS = MODELS

ATTENTION = Registry('attention', parent=MMCV_ATTENTION)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)

def build_mtlearner(cfg):
    return MTLEARNERS.build(cfg)

# def build_detector(cfg, train_cfg=None, test_cfg=None):
#     """Build detector."""
#     if train_cfg is not None or test_cfg is not None:
#         warnings.warn(
#             'train_cfg and test_cfg is deprecated, '
#             'please specify them in model', UserWarning)
#     assert cfg.get('train_cfg') is None or train_cfg is None, \
#         'train_cfg specified in both outer field and model field '
#     assert cfg.get('test_cfg') is None or test_cfg is None, \
#         'test_cfg specified in both outer field and model field '
#     return DETECTORS.build(
#         cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


# def build_segmentor(cfg, train_cfg=None, test_cfg=None):
#     """Build segmentor."""
#     if train_cfg is not None or test_cfg is not None:
#         warnings.warn(
#             'train_cfg and test_cfg is deprecated, '
#             'please specify them in model', UserWarning)
#     assert cfg.get('train_cfg') is None or train_cfg is None, \
#         'train_cfg specified in both outer field and model field '
#     assert cfg.get('test_cfg') is None or test_cfg is None, \
#         'test_cfg specified in both outer field and model field '
#     return SEGMENTORS.build(
#         cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

