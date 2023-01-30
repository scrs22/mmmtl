# Copyright (c) OpenMMLab. All rights reserved.
from mmmtl.models.builder import MTLEARNERS
from .detr import DETR


@MTLEARNERS.register_module()
class DeformableDETR(DETR):

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
