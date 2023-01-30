# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

from mmmtl.apis import inference_mtlearner, init_mtlearner, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the predict results by matplotlib.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_mtlearner(args.config, args.checkpoint, device=args.device,task="detection")
    # test a single image
    result = inference_mtlearner(model, args.img,task="detection")
    # show the results
    print(mmcv.dump(result, file_format='json', indent=4))
    if args.show:
        show_result_pyplot(model, args.img, result,task="detection")


if __name__ == '__main__':
    main()
