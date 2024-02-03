# -*- encoding: utf-8 -*-
"""
@File    :   nms.py
@Time    :   2023/12/21 22:43:42
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from ppdet.apis import manager
import ppdet.boxes.utils as box_utils


__all__ = ["MultiClassNMS"]


@manager.NMS.add_component
class MultiClassNMS:
    """
    Multi-class NMS
    """

    def __init__(
        self,
        use_multi_class_nms=True,
        nms_thresh=0.2,
        score_thresh=None,
        nms_pre_max_size=None,
        nms_post_max_size=None,
    ):

        super().__init__()

        self.use_multi_class_nms = use_multi_class_nms
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.nms_pre_max_size = nms_pre_max_size
        self.nms_post_max_size = nms_post_max_size

    def __call__(self, preds):
        """
        Args:
            preds (list[dict]): predictions

        Returns:
            results (list[dict]): results after nms
        """
        results = []
        for pred in preds:
            pred_boxes = pred["pred_boxes"]
            pred_scores = pred["pred_scores"]
            pred_labels = pred["pred_labels"]
            if len(pred_scores) == 0:
                result = dict(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                )
                results.append(result)
                continue

            # filter before nms
            indices = paddle.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[indices].reshape([-1, 4])
            pred_scores = pred_scores[indices]
            pred_labels = pred_labels[indices]
            max_num = len(pred_scores)
            if self.score_thresh is not None:
                max_num = min(
                    max_num, (pred_scores >= self.score_thresh).sum()
                )

            if self.nms_pre_max_size is not None:
                max_num = min(max_num, self.nms_pre_max_size)

            pred_boxes = pred_boxes[:max_num]
            pred_scores = pred_scores[:max_num]
            pred_labels = pred_labels[:max_num]
            if len(pred_scores) == 0:
                result = dict(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                )
                results.append(result)
                continue

            # nms
            boxes = box_utils.xywh2xyxy(pred_boxes)
            if self.use_multi_class_nms:
                categories = paddle.unique(pred_labels)
                indices = paddle.vision.ops.nms(
                    boxes,
                    iou_threshold=self.nms_thresh,
                    category_idxs=pred_labels,
                    categories=categories,
                    top_k=self.nms_post_max_size,
                )
            else:
                indices = paddle.vision.ops.nms(
                    boxes,
                    iou_threshold=self.nms_thresh,
                    top_k=self.nms_post_max_size,
                )

            pred_boxes = pred_boxes[indices].reshape([-1, 4])
            pred_scores = pred_scores[indices]
            pred_labels = pred_labels[indices]

            result = dict(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
            )
            results.append(result)

        return results
