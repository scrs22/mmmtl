_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/cub_bs64_224.py',
    '../_base_/schedules/cub_bs64.py',
    '../_base_/default_runtime.py'
]
