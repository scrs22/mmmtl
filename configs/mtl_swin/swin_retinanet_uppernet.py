_base_ = [
    # '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/imagenet_coco_ade20k_swin.py',
    '../_base_/schedules/imgnet_ade20k_coco_SGD.py', 
    '../_base_/default_runtime.py'
]


optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=50)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
optimizer_config = dict()

# mmcls
evaluation = dict(interval=1, metric='accuracy')
# mmseg
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
# mmdet
evaluation = dict(interval=1, metric='bbox')

# model settings
model = dict(
    type='ImageMTLearner',
    backbone=dict(
                # _delete_=True,
                type='SwinTransformerDet',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=(1, 2, 3),
                # Please only add indices that would be used
                # in FPN, otherwise some parameter will not be used
                with_cp=False,
                convert_weights=True,
                init_cfg=[
                    dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                    dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
                ],
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN', requires_grad=True)
        ),
    models=[
        dict(
            type='ImageClassifier',
            # backbone=dict(
            #     type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=768,
                init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                cal_acc=False),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ],
            train_cfg=dict(augments=[
                dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
                dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
            ])
        ),
        dict(
            type='RetinaNet',
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_input',
                num_outs=5),
            bbox_head=dict(
                type='RetinaHead',
                num_classes=80,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            # model training and testing settings
            train_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)
        ),
        dict(
            type='EncoderDecoder',
            pretrained=None,
            decode_head=dict(
                type='UPerHead',
                in_channels=[96, 192, 384, 768],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=384,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
            ),

        ]
)


# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# work_dir = '/nobackup/users/zitian/work_dirs/multi_learning'


# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',
#         embed_dims=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.2,
#         patch_norm=True,
#         out_indices=(1, 2, 3),
#         # Please only add indices that would be used
#         # in FPN, otherwise some parameter will not be used
#         with_cp=False,
#         convert_weights=True,
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5))

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
