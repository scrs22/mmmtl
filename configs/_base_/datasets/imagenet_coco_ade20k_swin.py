_base_ = ['./pipelines/rand_aug.py']

# dataset settings
cls_dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

cls_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

cls_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
# cls_data = dict(
#     samples_per_gpu=64,
#     workers_per_gpu=8,
#     train=dict(
#         type=cls_dataset_type,
#         data_prefix='data/imagenet/train',
#         pipeline=cls_train_pipeline),
#     val=dict(
#         type=cls_dataset_type,
#         data_prefix='data/imagenet/val',
#         # ann_file='data/imagenet/meta/val.txt',
#         pipeline=cls_test_pipeline),
#     test=dict(
#         # replace `data/val` with `data/test` for standard test
#         type=cls_dataset_type,
#         data_prefix='data/imagenet/val',
#         # ann_file='data/imagenet/meta/val.txt',
#         pipeline=cls_test_pipeline))

# evaluation = dict(interval=10, metric='accuracy')


# dataset settings
det_dataset_type = 'CocoDataset'
det_data_root = 'data/coco/'
det_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
det_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# det_data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=det_dataset_type,
#         ann_file=det_data_root + 'annotations/instances_train2017.json',
#         img_prefix=det_data_root + 'train2017/',
#         pipeline=det_train_pipeline),
#     val=dict(
#         type=det_dataset_type,
#         ann_file=det_data_root + 'annotations/instances_val2017.json',
#         img_prefix=det_data_root + 'val2017/',
#         pipeline=det_test_pipeline),
#     test=dict(
#         type=det_dataset_type,
#         ann_file=det_data_root + 'annotations/instances_val2017.json',
#         img_prefix=det_data_root + 'val2017/',
#         pipeline=det_test_pipeline))
# evaluation = dict(metric=['bbox', 'segm'])


# dataset settings
seg_dataset_type = 'ADE20KDataset'
seg_data_root = 'data/ADEChallengeData2016'
crop_size = (512, 512)
seg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
seg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# seg_data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=4,
#     train=dict(
#         type=seg_dataset_type,
#         data_root=seg_data_root,
#         img_dir='images/training',
#         ann_dir='annotations/training',
#         pipeline=seg_train_pipeline),
#     val=dict(
#         type=seg_dataset_type,
#         data_root=seg_data_root,
#         img_dir='images/validation',
#         ann_dir='annotations/validation',
#         pipeline=seg_est_pipeline),
#     test=dict(
#         type=seg_dataset_type,
#         data_root=seg_data_root,
#         img_dir='images/validation',
#         ann_dir='annotations/validation',
#         pipeline=seg_test_pipeline))

cat_batchsizes = [1,1,1]  # cls, det, seg
cat_weights = [0.4,0.3,0.3]
cat_epochsize = 32
cat_shuffle = True
cat_dataloader = dict(samples_per_gpu=8, workers_per_gpu=2,
    sampler_cfg=dict(
        type='Distributed_Weighted_BatchSampler', 
        batch_sizes = cat_batchsizes,
        weights = cat_weights, 
        num_samples = cat_epochsize,
        shuffle = cat_shuffle
    )
)

data = dict(
    # samples_per_gpu=4,
    # workers_per_gpu=4,
    train=dict(
        type='ConcatMultiTypeDataset',
        datasets = [
            dict(type=cls_dataset_type,data_prefix='data/imagenet/train',pipeline=cls_train_pipeline),
            dict(type=det_dataset_type,ann_file=det_data_root + 'annotations/instances_train2017.json',img_prefix=det_data_root + 'train2017/',pipeline=det_train_pipeline),
            dict(type=seg_dataset_type,data_root=seg_data_root,img_dir='images/training',ann_dir='annotations/training',pipeline=seg_train_pipeline),
        ]
    ),
    train_dataloader=cat_dataloader,
    val=dict(
        type='ConcatMultiTypeDataset',
        datasets = [
            dict(type=cls_dataset_type,data_prefix='data/imagenet/val',pipeline=cls_test_pipeline),
            dict(type=det_dataset_type,ann_file=det_data_root + 'annotations/instances_val2017.json',img_prefix=det_data_root + 'val2017/',pipeline=det_test_pipeline),
            dict(type=seg_dataset_type,data_root=seg_data_root,img_dir='images/validation',ann_dir='annotations/validation',pipeline=seg_test_pipeline),
        ]
    ),
    val_dataloader=cat_dataloader,
    test=dict(
        type='ConcatMultiTypeDataset',
        datasets = [
            dict(type=cls_dataset_type,data_prefix='data/imagenet/val',pipeline=cls_test_pipeline),
            dict(type=det_dataset_type,ann_file=det_data_root + 'annotations/instances_val2017.json',img_prefix=det_data_root + 'val2017/',pipeline=det_test_pipeline),
            dict(type=seg_dataset_type,data_root=seg_data_root,img_dir='images/validation',ann_dir='annotations/validation',pipeline=seg_test_pipeline),
        ]
    ),
    test_dataloader=cat_dataloader
)

