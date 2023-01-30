# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
import mmcv
from mmcv.utils import build_from_cfg
from mmcv.runner import (HOOKS,DistSamplerSeedHook, EpochBasedRunner,build_optimizer,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)

from mmmtl.core import DistEvalHookCls, DistOptimizerHook, EvalHookCls,EvalHookDet,DistEvalHookDet, EvalHookSeg,DistEvalHookSeg, build_optimizer
from mmmtl.datasets import build_dataloader, build_dataset
from mmmtl.datasets.det.utils import replace_ImageToTensor
from mmmtl.utils import (auto_select_device, get_root_logger,
                         wrap_distributed_model, wrap_non_distributed_model,build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint)
from mmmtl.utils.tasks import * 
from mmmtl import digit_version


def init_random_seed(seed=None, device=None):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed
    if device is None:
        device = auto_select_device()
    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmmtlection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')

def train_classifier(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                device=None,
                meta=None):
    """Train a model.

    This method will build dataloaders, wrap the model and build a runner
    according to the provided config.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        dataset (:obj:`mmmtl.datasets.BaseDataset` | List[BaseDataset]):
            The dataset used to train the model. It can be a single dataset,
            or a list of dataset with the same length as workflow.
        cfg (:obj:`mmcv.utils.Config`): The configs of the experiment.
        distributed (bool): Whether to train the model in a distributed
            environment. Defaults to False.
        validate (bool): Whether to do validation with
            :obj:`mmcv.runner.EvalHook`. Defaults to False.
        timestamp (str, optional): The timestamp string to auto generate the
            name of log files. Defaults to None.
        device (str, optional): TODO
        meta (dict, optional): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
    """
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=cfg.ipu_replicas if device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}

    data_loaders = [build_dataloader(ds, **train_loader_cfg, task=CLASSIFICATION) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = wrap_distributed_model(
            model,
            cfg.device,
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = wrap_non_distributed_model(
            model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    if device == 'ipu':
        if not cfg.runner['type'].startswith('IPU'):
            cfg.runner['type'] = 'IPU' + cfg.runner['type']
        if 'options_cfg' not in cfg.runner:
            cfg.runner['options_cfg'] = {}
        cfg.runner['options_cfg']['replicationFactor'] = cfg.ipu_replicas
        cfg.runner['fp16_cfg'] = cfg.get('fp16', None)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)

    if fp16_cfg is None and device == 'npu':
        fp16_cfg = {'loss_scale': 'dynamic'}

    if fp16_cfg is not None:
        if device == 'ipu':
            from mmcv.device.ipu import IPUFp16OptimizerHook
            optimizer_config = IPUFp16OptimizerHook(
                **cfg.optimizer_config,
                loss_scale=fp16_cfg['loss_scale'],
                distributed=distributed)
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config,
                loss_scale=fp16_cfg['loss_scale'],
                distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed and cfg.runner['type'] == 'EpochBasedRunner':
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True),task=CLASSIFICATION)
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'shuffle': False,  # Not shuffle by default
            'sampler_cfg': None,  # Not use sampler by default
            'drop_last': False,  # Not drop last by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg,task=CLASSIFICATION)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHookCls if distributed else EvalHookCls
        # `EvalHook` needs to be executed after `IterTimerHook`.
        # Otherwise, it will cause a bug if use `IterBasedRunner`.
        # Refers to https://github.com/open-mmlab/mmcv/issues/1261
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    data_loaders = [build_dataloader(ds, **train_loader_cfg,task=SEGMENTATION) for ds in dataset]

    # put model on devices
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # DDP wrapper
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        # when distributed training by epoch, using`DistSamplerSeedHook` to set
        # the different seed to distributed sampler for each epoch, it will
        # shuffle dataset at each epoch and avoid overfitting.
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True),task=SEGMENTATION)
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg,task=SEGMENTATION)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHookSeg if distributed else EvalHookSeg
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg,task=DETECTION) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True),task=DETECTION)

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args,task=DETECTION)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHookDet if distributed else EvalHookDet
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def train_mtlearner(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                device=None,
                meta=None,
                task=DETECTION):
    """Train a model.

    This method will build dataloaders, wrap the model and build a runner
    according to the provided config.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        dataset (:obj:`mmmtl.datasets.BaseDataset` | List[BaseDataset]):
            The dataset used to train the model. It can be a single dataset,
            or a list of dataset with the same length as workflow.
        cfg (:obj:`mmcv.utils.Config`): The configs of the experiment.
        distributed (bool): Whether to train the model in a distributed
            environment. Defaults to False.
        validate (bool): Whether to do validation with
            :obj:`mmcv.runner.EvalHook`. Defaults to False.
        timestamp (str, optional): The timestamp string to auto generate the
            name of log files. Defaults to None.
        device (str, optional): TODO
        meta (dict, optional): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
    """
    if task==CLASSIFICATION:
        return train_classifier(model,
                dataset,
                cfg,
                distributed,
                validate,
                timestamp,
                device,
                meta)
    elif task==SEGMENTATION:
        train_segmentor(model,
                    dataset,
                    cfg,
                    distributed,
                    validate,
                    timestamp,
                    meta)
    else:
        return train_detector(model,
                   dataset,
                   cfg,
                   distributed,
                   validate,
                   timestamp,
                   meta)
