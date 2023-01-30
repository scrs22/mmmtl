# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm

from mmmtl.apis import inference_mtlearner, init_mtlearner, show_result_pyplot
from mmmtl.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--task',
        
        default="detection",
        help='task')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_mtlearner(args.config, args.checkpoint, device=args.device,task=args.task)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_mtlearner(model, args.img,task=args.task)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        opacity=args.opacity,
        out_file=args.out_file,
        task=args.task)


if __name__ == '__main__':
    main()
