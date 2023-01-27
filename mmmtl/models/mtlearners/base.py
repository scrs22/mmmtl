# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Sequence

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmmtl.core.visualization import imshow_infos
from mmmtl.model.mtlearners.classifiers import BaseClassifier
from mmmtl.model.mtlearners.segmentors import BaseSegmentor
from mmmtl.model.mtlearners.detectors import BaseDetector

class BaseMTLearner(BaseDetector,BaseSegmentor,BaseClassifier,metaclass=ABCMeta):
    """Base class for MTLearners."""

    def __init__(self, init_cfg=None):
        super(BaseMTLearner, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @abstractmethod
    def extract_feat(self, imgs, stage=None):
        pass

    def extract_feats(self, imgs, stage=None):
        assert isinstance(imgs, Sequence)
        kwargs = {} if stage is None else {'stage': stage}
        for img in imgs:
            yield self.extract_feat(img, **kwargs)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmmtl.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        if not img_metas or len(img_meta)==0:
            pass
        batch_input_shape = tuple(imgs[0].size()[-2:])
        
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    def forward_test(self, imgs, img_metas=None, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_meta are single-nested (i.e. Tensor and
        List[dict]), and when `resturn_loss=False`, img and img_meta should be
        double nested (i.e.  List[Tensor], List[List[dict]]), with the outer
        list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def show_result(self,
                    img,
                    result,
                    text_color='white',
                    font_scale=0.5,
                    row_width=20,
                    show=False,
                    fig_size=(15, 10),
                    win_name='',
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or ndarray): The image to be displayed.
            result (dict): The classification results to draw over `img`.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            font_scale (float): Font scales of texts.
            row_width (int): width between each row of results on the image.
            show (bool): Whether to show the image.
                Default: False.
            fig_size (tuple): Image show figure size. Defaults to (15, 10).
            win_name (str): The window name.
            wait_time (int): How many seconds to display the image.
                Defaults to 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (ndarray): Image with overlaid results.
        """
        img = mmcv.imread(img)
        img = img.copy()

        img = imshow_infos(
            img,
            result,
            text_color=text_color,
            font_size=int(font_scale * 50),
            row_width=row_width,
            win_name=win_name,
            show=show,
            fig_size=fig_size,
            wait_time=wait_time,
            out_file=out_file)

        return img
