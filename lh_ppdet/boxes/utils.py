# -*- encoding: utf-8 -*-
"""
@File    :   box_utils.py
@Time    :   2023/12/18 14:00:11
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle


def xywh2xyxy(boxes):
    """
    Convert [x, y, w, h] box format to [x1, y1, x2, y2] format

    Args:
        boxes (Tensor): boxes with shape [..., N, 4], format is [x, y, w, h]

    Returns:
        boxes (Tensor): boxes with shape [..., N, 4], format is [x1, y1, x2, y2]
    """
    x1y1 = boxes[..., :2] - boxes[..., 2:] / 2.0
    x2y2 = boxes[..., :2] + boxes[..., 2:] / 2.0
    boxes = paddle.concat([x1y1, x2y2], axis=-1)

    return boxes


def xyxy2xywh(boxes):
    """
    Convert [x1, y1, x2, y2] box format to [x, y, w, h] format

    Args:
        boxes (Tensor): boxes with shape [..., N, 4], format is [x1, y1, x2, y2]

    Returns:
        boxes (Tensor): boxes with shape [..., N, 4], format is [x, y, w, h]
    """
    xy = (boxes[..., :2] + boxes[..., 2:]) / 2.0
    wh = boxes[..., 2:] - boxes[..., :2]
    boxes = paddle.concat([xy, wh], axis=-1)

    return boxes


def box_iou(boxes1, boxes2):
    """calculate box iou"""
    x1_min, y1_min, x1_max, y1_max = paddle.unstack(xywh2xyxy(boxes1), axis=-1)
    s1 = boxes1[..., 2] * boxes1[..., 3]

    x2_min, y2_min, x2_max, y2_max = paddle.unstack(xywh2xyxy(boxes2), axis=-1)
    s2 = boxes2[..., 2] * boxes2[..., 3]

    xmin = paddle.maximum(x1_min, x2_min)
    ymin = paddle.maximum(y1_min, y2_min)
    xmax = paddle.minimum(x1_max, x2_max)
    ymax = paddle.minimum(y1_max, y2_max)
    inter_h = paddle.clip(ymax - ymin, 0.0)
    inter_w = paddle.clip(xmax - xmin, 0.0)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    ious = intersection / union

    return ious


class IouSimilarity:
    """
    IoU Similarity
    """

    def __init__(
        self,
        is_aligned=False,
    ):
        self.is_aligned = is_aligned

    def __call__(self, boxes1, boxes2):
        """
        Calculate IoU between boxes1 and boxes2

        Args:
            boxes1 (Tensor): boxes1 with shape [..., N, 4]
            boxes2 (Tensor): boxes2 with shape [..., M, 4] or [..., N, 4] if is_aligned=True

        Returns:
            ious (Tensor): IoU between boxes1 and boxes2 with shape [..., N, M] or [..., N] if is_aligned=True
        """
        if self.is_aligned:
            assert (
                boxes1.shape[-2] == boxes2.shape[-2]
            ), "boxes1 and boxes2 must have same number"
            return box_iou(boxes1, boxes2)
        else:
            return box_iou(
                boxes1.unsqueeze(axis=-2), boxes2.unsqueeze(axis=-3)
            )
