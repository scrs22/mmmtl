# Copyright (c) OpenMMLab. All rights reserved.
from .lamb import Lamb

cls = [
    'Lamb',
]


# Copyright (c) OpenMMLab. All rights reserved.
from .builder import OPTIMIZER_BUILDERS, build_optimizer
from .layer_decay_optimizer_constructor import \
    LearningRateDecayOptimizerConstructor

det = [
    'LearningRateDecayOptimizerConstructor', 'OPTIMIZER_BUILDERS',
    'build_optimizer'
]



# Copyright (c) OpenMMLab. All rights reserved.
from .layer_decay_optimizer_constructor import (
    LayerDecayOptimizerConstructor, LearningRateDecayOptimizerConstructor)

seg = [
    'LearningRateDecayOptimizerConstructor', 'LayerDecayOptimizerConstructor'
]


__all__ = list(set(cls) | set(det) | set(seg))
