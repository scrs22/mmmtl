# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseMTLearner
from .image import ImageMTLearner
from mmmtl.models.classifiers import *
from mmmtl.models.detectors import *
from mmmtl.models.segmentors import *

__all__ = ['BaseMTLearner']
