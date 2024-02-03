# -*- encoding: utf-8 -*-
"""
@File    :   box_coder.py
@Time    :   2023/12/18 14:26:41
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn.functional as F
from ppdet.apis import manager
from ppdet.boxes import IouSimilarity


@manager.BOX_CODERS.add_component
class YOLOv3BoxCoder:
    """
    YOLOv3 Box Coder
    """

    def __init__(
        self,
        num_classes,
        image_size,
        iou_aware=False,
        iou_aware_factor=0.4,
        conf_thresh=None,
        max_num=None,
    ):
        self.num_classes = num_classes
        self.image_size = image_size

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.conf_thresh = conf_thresh
        self.max_num = max_num

        self.iou_calculator = IouSimilarity()

    def encode(self, gt_boxes, gt_labels, anchors, downsample):
        """
        encode

        Args:
            gt_boxes (Tensor): ground truth boxes, shape [B, N, 4]
            gt_labels (Tensor): ground truth labels, shape [B, N]
            anchors (Tensor): list of anchors, shape [num_anchors, 2]
            downsample (int): downsample ratio

        Returns:
            target_dict (dict): target dict
        """
        B = len(gt_boxes)
        num_anchors = len(anchors)
        anchors = paddle.to_tensor(anchors, dtype=gt_boxes.dtype)

        assert (
            self.image_size[0] % downsample == 0
            and self.image_size[1] % downsample == 0
        ), "image size must be divided by downsample"
        feat_size = (
            self.image_size[0] // downsample,
            self.image_size[1] // downsample,
        )
        conf_target = paddle.zeros(
            (B, feat_size[1], feat_size[0], num_anchors),
            dtype=gt_boxes.dtype,
        )
        box_target = paddle.zeros(
            (B, feat_size[1], feat_size[0], num_anchors, 4),
            dtype=gt_boxes.dtype,
        )
        cls_target = paddle.zeros(
            (B, feat_size[1], feat_size[0], num_anchors, self.num_classes),
            dtype=gt_boxes.dtype,
        )
        box_weight = paddle.zeros(
            (B, feat_size[1], feat_size[0], num_anchors),
            dtype=gt_boxes.dtype,
        )

        for batch_id in range(B):
            cur_gt_boxes = gt_boxes[batch_id]
            cur_gt_labels = gt_labels[batch_id]
            mask = cur_gt_labels >= 0
            cur_gt_boxes = paddle.masked_select(
                cur_gt_boxes, mask[:, None].tile([1, 4])
            ).reshape([-1, 4])
            cur_gt_labels = paddle.masked_select(cur_gt_labels, mask)
            if len(cur_gt_boxes) == 0:
                continue

            # normalize boxes
            center_x = cur_gt_boxes[:, 0] / downsample
            center_y = cur_gt_boxes[:, 1] / downsample

            # calculate iou
            boxes = paddle.zeros_like(cur_gt_boxes)
            boxes[:, 2] = cur_gt_boxes[:, 2] / self.image_size[0]
            boxes[:, 3] = cur_gt_boxes[:, 3] / self.image_size[1]
            anchor_boxes = paddle.zeros((num_anchors, 4), dtype=anchors.dtype)
            anchor_boxes[:, 2] = anchors[:, 0] / self.image_size[0]
            anchor_boxes[:, 3] = anchors[:, 1] / self.image_size[1]
            ious = self.iou_calculator(boxes, anchor_boxes)
            ks = paddle.argmax(ious, axis=1)

            xs = center_x.astype("int32").clip(0, feat_size[0] - 1)
            ys = center_y.astype("int32").clip(0, feat_size[1] - 1)
            bs = paddle.full_like(xs, batch_id)
            cs = cur_gt_labels

            dx = center_x - xs
            dy = center_y - ys
            dw = paddle.log(cur_gt_boxes[:, 2] / anchors[:, 0][ks])
            dh = paddle.log(cur_gt_boxes[:, 3] / anchors[:, 1][ks])

            conf_target[(bs, ys, xs, ks)] = 1
            box_target[(bs, ys, xs, ks)] = paddle.stack(
                [dx, dy, dw, dh], axis=-1
            )
            cls_target[(bs, ys, xs, ks, cs)] = 1
            box_weight[(bs, ys, xs, ks)] = 2 - cur_gt_boxes[
                :, 2
            ] * cur_gt_boxes[:, 3] / (self.image_size[0] * self.image_size[1])

        target_dict = dict(
            conf_target=conf_target[..., None],
            box_target=box_target,
            cls_target=cls_target,
            box_weight=box_weight[..., None],
        )

        return target_dict

    def decode(self, output, anchors, downsample, filter=False) -> list:
        """
        decode

        Args:
            output (Tensor): output of network, shape: [B, C, H, W]
            anchors (list): anchors, shape: [num_anchors, 2]
            downsample (int): downsample ratio
            filter (bool): whether to filter the results

        Returns:
            results (dict|list): dict or list of boxes, scores and labels
        """
        # output shape: [B, C, H, W], C = num_anchors * (box_dim + obj_dim + num_classes + iou_dim)
        B, C, H, W = output.shape
        num_anchors = len(anchors)
        output = output.transpose([0, 2, 3, 1]).reshape(
            [B, H, W, num_anchors, -1]
        )

        # box
        box = output[..., :4].reshape([B, -1, 4])
        anchor_x = paddle.arange(W, dtype=output.dtype)
        anchor_y = paddle.arange(H, dtype=output.dtype)
        anchor_y, anchor_x = paddle.meshgrid(anchor_y, anchor_x)
        anchor_x = anchor_x.reshape([-1, 1, 1]).tile([1, num_anchors, 1])
        anchor_y = anchor_y.reshape([-1, 1, 1]).tile([1, num_anchors, 1])
        anchor_wh = paddle.to_tensor(anchors, output.dtype)
        anchor_wh = paddle.tile(anchor_wh, [H * W, 1, 1])
        anchors = paddle.concat(
            [anchor_x, anchor_y, anchor_wh], axis=-1
        )  # [H * W, num_anchors, 4]
        anchors = anchors.reshape([-1, 4])

        pred_boxes = paddle.zeros_like(box)
        pred_boxes[..., :2] = (
            F.sigmoid(box[..., :2]) + anchors[..., :2]
        ) * downsample
        pred_boxes[..., 2:] = paddle.exp(box[..., 2:]) * anchors[..., 2:]

        # confidence
        confidence = F.sigmoid(output[..., 4]).reshape([B, -1])
        if self.iou_aware:
            iou_score = output[..., -1].reshape([B, -1])
            confidence = iou_score**self.iou_aware_factor + confidence ** (
                1 - self.iou_aware_factor
            )

        # class
        score = F.sigmoid(output[..., 5 : 5 + self.num_classes]).reshape(
            [B, -1, self.num_classes]
        )
        pred_scores = paddle.max(score, axis=-1) * confidence
        pred_labels = paddle.argmax(score, axis=-1)

        # filter
        if filter:
            results = []
            for batch_id in range(B):
                cur_pred_boxes = pred_boxes[batch_id]
                cur_pred_scores = pred_scores[batch_id]
                cur_pred_labels = pred_labels[batch_id]
                cur_confidence = confidence[batch_id]

                indices = paddle.argsort(cur_pred_scores, descending=True)
                cur_pred_boxes = cur_pred_boxes[indices].reshape([-1, 4])
                cur_pred_scores = cur_pred_scores[indices]
                cur_pred_labels = cur_pred_labels[indices]
                max_num = len(pred_scores)
                if self.conf_thresh is not None:
                    max_num = min(
                        max_num, (cur_confidence >= self.conf_thresh).sum()
                    )

                if self.max_num is not None:
                    max_num = min(max_num, self.max_num)

                cur_pred_boxes = cur_pred_boxes[:max_num]
                cur_pred_scores = cur_pred_scores[:max_num]
                cur_pred_labels = cur_pred_labels[:max_num]
                result = dict(
                    pred_boxes=cur_pred_boxes,
                    pred_scores=cur_pred_scores,
                    pred_labels=cur_pred_labels,
                )
                results.append(result)
        else:
            results = dict(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
            )

        return results
