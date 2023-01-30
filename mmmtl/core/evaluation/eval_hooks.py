# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
# Copyright (c) OpenMMLab. All rights reserved.
import bisect

import mmcv

class EvalHookCls(BaseEvalHook):
    """Non-Distributed evaluation hook.

    Comparing with the ``EvalHook`` in MMCV, this hook will save the latest
    evaluation results as an attribute for other hooks to use (like
    `MMClsWandbHook`).
    """

    def __init__(self, dataloader, **kwargs):
        super(EvalHookCls, self).__init__(dataloader, **kwargs)
        self.latest_results = None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        results = self.test_fn(runner.model, self.dataloader)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


class DistEvalHookCls(BaseDistEvalHook):
    """Non-Distributed evaluation hook.

    Comparing with the ``EvalHook`` in MMCV, this hook will save the latest
    evaluation results as an attribute for other hooks to use (like
    `MMClsWandbHook`).
    """

    def __init__(self, dataloader, **kwargs):
        super(DistEvalHookCls, self).__init__(dataloader, **kwargs)
        self.latest_results = None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            # the key_score may be `None` so it needs to skip the action to
            # save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)

def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class EvalHookDet(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(EvalHookDet, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmmtl.apis import single_gpu_test

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.
class DistEvalHookDet(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(DistEvalHookDet, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmmtl.apis import multi_gpu_test

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)


class EvalHookSeg(BaseEvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None

        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmmtl.apis import single_gpu_test
        results = single_gpu_test(
            runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHookSeg(BaseDistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmmtl.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
