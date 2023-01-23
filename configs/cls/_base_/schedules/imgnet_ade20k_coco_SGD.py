
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=50)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
optimizer_config = dict()

# # mmcls
# paramwise_cfg = dict(
#     norm_decay_mult=0.0,
#     bias_decay_mult=0.0,
#     custom_keys={
#         '.absolute_pos_embed': dict(decay_mult=0.0),
#         '.relative_position_bias_table': dict(decay_mult=0.0)
#     })

# # for batch in each gpu is 128, 8 gpu
# # lr = 5e-4 * 128 * 8 / 512 = 0.001
# optimizer = dict(
#     type='AdamW',
#     lr=5e-4 * 1536 / 512,
#     weight_decay=0.05,
#     eps=1e-8,
#     betas=(0.9, 0.999),
#     paramwise_cfg=paramwise_cfg)
# optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# # learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     min_lr_ratio=1e-2,
#     warmup='linear',
#     warmup_ratio=1e-3,
#     warmup_iters=20,
#     warmup_by_epoch=True)

# runner = dict(type='EpochBasedRunner', max_epochs=300)

# # mmseg
# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=160000)
# checkpoint_config = dict(by_epoch=False, interval=16000)
# evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)


# # mmdet
# # optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)

