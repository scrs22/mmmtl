# Copyright (c) OpenMMLab. All rights reserved.
from mmmtl.models.builder import MTLEARNERS
from .cascade_rcnn import CascadeRCNN


@MTLEARNERS.register_module()
class SCNet(CascadeRCNN):
    """Implementation of `SCNet <https://arxiv.org/abs/2012.10150>`_"""

    def __init__(self, **kwargs):
        super(SCNet, self).__init__(**kwargs)
