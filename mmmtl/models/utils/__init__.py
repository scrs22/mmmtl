# Copyright (c) OpenMMLab. All rights reserved.
from .attention import MultiheadAttention, ShiftWindowMSA, WindowMSAV2
from .augment.augments import Augments
from .channel_shuffle import channel_shuffle
from .embed import (HybridEmbed, PatchEmbed, PatchMerging, resize_pos_embed,
                    resize_relative_position_bias_table)
from .helpers import is_tracing, to_2tuple, to_3tuple, to_4tuple, to_ntuple
from .inverted_residual import InvertedResidual
from .layer_scale import LayerScale
from .make_divisible import make_divisible
from .position_encoding import ConditionalPositionEncoding
from .se_layer import SELayer




# Copyright (c) OpenMMLab. All rights reserved.
from .brick_wrappers import AdaptiveAvgPool2d, adaptive_avg_pool2d
from .builder import build_linear_layer, build_transformer
from .ckpt_convert import pvt_convert
from .conv_upsample import ConvUpsample
from .csp_layer import CSPLayer
from .gaussian_target import gaussian_radius, gen_gaussian_target
from .inverted_residual import InvertedResidual
from .make_divisible import make_divisible
from .misc import interpolate_as, sigmoid_geometric_mean
from .normed_predictor import  NormedLinear
from .panoptic_gt_processing import preprocess_panoptic_gt
from .point_sample import (get_uncertain_point_coords_with_randomness,
                           get_uncertainty)
# from .positional_encoding import (LearnedPositionalEncoding)
from .res_layer import ResLayer, SimplifiedBasicBlock
from .se_layer import DyReLU, SELayer
# from .transformer import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
#                           DynamicConv, PatchEmbed, Transformer, nchw_to_nlc,
#                           nlc_to_nchw)
from .transformer import PatchEmbed, nchw_to_nlc, nlc_to_nchw

# __all__ = [
#     'channel_shuffle', 'make_divisible', 'InvertedResidual', 'SELayer',
#     'to_ntuple', 'to_2tuple', 'to_3tuple', 'to_4tuple', 'PatchEmbed',
#     'PatchMerging', 'HybridEmbed', 'Augments', 'ShiftWindowMSA', 'is_tracing',
#     'MultiheadAttention', 'ConditionalPositionEncoding', 'resize_pos_embed',
#     'resize_relative_position_bias_table', 'WindowMSAV2', 'LayerScale'
# ]

det = [
'channel_shuffle', 'make_divisible', 'InvertedResidual', 'SELayer',
    'to_ntuple', 'to_2tuple', 'to_3tuple', 'to_4tuple', 'PatchEmbed',
    'PatchMerging', 'HybridEmbed', 'Augments', 'ShiftWindowMSA', 'is_tracing',
    'MultiheadAttention', 'ConditionalPositionEncoding', 'resize_pos_embed',
    'resize_relative_position_bias_table', 'WindowMSAV2', 'LayerScale',
    'ResLayer', 'gaussian_radius', 'gen_gaussian_target',
    # 'DetrTransformerDecoderLayer', 'DetrTransformerDecoder', 'Transformer', 'DynamicConv',
    'build_transformer', 'build_linear_layer',
    'LearnedPositionalEncoding',  'SimplifiedBasicBlock',
    'NormedLinear', 'make_divisible', 'InvertedResidual',
    'SELayer', 'interpolate_as', 'ConvUpsample', 'CSPLayer',
    'adaptive_avg_pool2d', 'AdaptiveAvgPool2d', 'PatchEmbed', 'nchw_to_nlc',
    'nlc_to_nchw', 'pvt_convert', 'sigmoid_geometric_mean',
    'preprocess_panoptic_gt', 'DyReLU',
    'get_uncertain_point_coords_with_randomness', 'get_uncertainty'
]


# Copyright (c) OpenMMLab. All rights reserved.
from .embed import PatchEmbed
from .inverted_residual import InvertedResidual, InvertedResidualV3
from .make_divisible import make_divisible
from .res_layer import ResLayer
from .se_layer import SELayer
from .self_attention_block import SelfAttentionBlock
from .shape_convert import (nchw2nlc2nchw, nchw_to_nlc, nlc2nchw2nlc,
                            nlc_to_nchw)
from .up_conv_block import UpConvBlock

seg = [
    'ResLayer', 'SelfAttentionBlock', 'make_divisible', 'InvertedResidual',
    'UpConvBlock', 'InvertedResidualV3', 'SELayer', 'PatchEmbed',
    'nchw_to_nlc', 'nlc_to_nchw', 'nchw2nlc2nchw', 'nlc2nchw2nlc'
]

__all__ = list(set(det) | set(seg))
