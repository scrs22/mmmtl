# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseMTLearner
from mmmtl.models.mtlearners.classifiers import *
from mmmtl.models.mtlearners.detectors import *
from mmmtl.models.mtlearners.segmentors import *

__all__ = ['BaseMTLearner']
