# Copyright (c) OpenMMLab. All rights reserved.
from .eval_hooks import DistEvalHook, EvalHook
from .eval_metrics import (calculate_confusion_matrix, f1_score, precision,
                           precision_recall_f1, recall, support)
from .mean_ap import average_precision, mAP
from .multilabel_eval_metrics import average_performance

cls = [
    'precision', 'recall', 'f1_score', 'support', 'average_precision', 'mAP',
    'average_performance', 'calculate_confusion_matrix', 'precision_recall_f1',
    'EvalHook', 'DistEvalHook'
]


# Copyright (c) OpenMMLab. All rights reserved.
from .class_names import (cityscapes_classes, coco_classes, dataset_aliases,
                          get_classes, imagenet_det_classes,
                          imagenet_vid_classes, oid_challenge_classes,
                          oid_v6_classes, voc_classes)
from .eval_hooks import DistEvalHook, EvalHook
from .mean_ap import average_precision, eval_map, print_map_summary
from .panoptic_utils import INSTANCE_OFFSET
from .recall import (eval_recalls, plot_iou_recall, plot_num_recall,
                     print_recall_summary)

det = [
    'voc_classes', 'imagenet_det_classes', 'imagenet_vid_classes',
    'coco_classes', 'cityscapes_classes', 'dataset_aliases', 'get_classes',
    'DistEvalHook', 'EvalHook', 'average_precision', 'eval_map',
    'print_map_summary', 'eval_recalls', 'print_recall_summary',
    'plot_num_recall', 'plot_iou_recall', 'oid_v6_classes',
    'oid_challenge_classes', 'INSTANCE_OFFSET'
]


# Copyright (c) OpenMMLab. All rights reserved.
from .class_names import get_classes, get_palette
from .eval_hooks import DistEvalHook, EvalHook
from .metrics import (eval_metrics, intersect_and_union, mean_dice,
                      mean_fscore, mean_iou, pre_eval_to_metrics)

seg = [
    'EvalHook', 'DistEvalHook', 'mean_dice', 'mean_iou', 'mean_fscore',
    'eval_metrics', 'get_classes', 'get_palette', 'pre_eval_to_metrics',
    'intersect_and_union'
]


__all__ = list(set(cls) | set(det) | set(seg))
