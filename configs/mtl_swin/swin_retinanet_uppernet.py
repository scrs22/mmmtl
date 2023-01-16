_base_ = [
    # '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/imagenet_coco_ade20k_swin.py',
    # '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa


# model settings
model = dict(
    type='',
    backbone=dict(
                _delete_=True,
                type='SwinTransformer',
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
                init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    models=[
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
        ]
)


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

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
