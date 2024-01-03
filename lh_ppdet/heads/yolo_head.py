# -*- encoding: utf-8 -*-
"""
@File    :   yolo_head.py
@Time    :   2023/12/16 15:53:45
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
from lh_ppdet.apis import manager
from lh_ppdet.boxes import IouSimilarity


__all__ = ["YOLOv3Head"]


@manager.HEADS.add_component
class YOLOv3Head(nn.Layer):
    """
    Head for YOLOv3 network
    """

    def __init__(
        self,
        in_channels,
        anchors=[
            [10, 13],
            [16, 30],
            [33, 23],
            [30, 61],
            [62, 45],
            [59, 119],
            [116, 90],
            [156, 198],
            [373, 326],
        ],
        anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        downsamples=[8, 16, 32],
        num_classes=80,
        ignore_thresh=0.4,
        loss_xy=None,
        loss_wh=None,
        loss_cls=None,
        loss_conf=None,
        loss_iou=None,
        loss_iou_aware=None,
        box_coder=None,
        nms=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_iou = loss_iou
        self.loss_iou_aware = loss_iou_aware
        self.iou_aware = loss_iou_aware is not None
        self.ignore_thresh = ignore_thresh

        self.loss_xy = loss_xy
        self.loss_wh = loss_wh
        self.loss_cls = loss_cls
        self.loss_conf = loss_conf
        self.box_coder = box_coder
        self.nms = nms
        self.iou_calculator = IouSimilarity()

        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.downsamples = downsamples
        assert len(self.anchors) == len(
            self.downsamples
        ), "anchor and downsample should have same length"

        self.heads = nn.LayerList()
        for i in range(len(self.anchors)):
            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            head = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
            )
            self.heads.append(head)

    def forward(self, feats, **kwargs) -> dict:
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        outs = [self.heads[i](feat) for i, feat in enumerate(feats)]

        pred_dict = dict(
            outputs=outs,
        )

        return pred_dict

    def loss(self, pred_dict, gt_boxes, gt_labels, **kwargs) -> dict:
        """
        YOLOv3 loss

        Args:
            pred_dict (dict): dict of predicted outputs
            gt_boxes (Tensor): ground truth boxes, shape [B, N, 4]
            gt_labels (Tensor): ground truth labels, shape [B, N]

        Returns:
            loss_dict (dict): dict of loss
        """
        outputs = pred_dict["outputs"]
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        gt_boxes = gt_boxes.astype(outputs[0].dtype)
        loss_dict = dict()

        # caclulate multi-scale loss
        for scale, (output, anchors, downsample) in enumerate(
            zip(outputs, self.anchors, self.downsamples)
        ):
            # get target
            target_dict = self.box_coder.encode(
                gt_boxes, gt_labels, anchors, downsample
            )
            box_target = target_dict["box_target"]
            box_weight = target_dict["box_weight"]
            cls_target = target_dict["cls_target"]
            conf_target = target_dict["conf_target"]

            # get output
            B, C, H, W = output.shape
            num_anchors = len(anchors)
            output = output.transpose([0, 2, 3, 1]).reshape(
                [B, H, W, num_anchors, -1]
            )

            pred_box = output[..., :4]
            pred_conf = output[..., 4:5]
            pred_cls = output[..., 5 : 5 + self.num_classes]
            if self.iou_aware:
                pred_iou = output[..., -1]

            # box loss
            avg_factor = max(1, conf_target.sum())
            loss_xy = self.loss_xy(
                pred_box[..., :2],
                box_target[..., :2],
                weight=box_weight,
                avg_factor=avg_factor,
            )
            loss_wh = self.loss_wh(
                pred_box[..., 2:],
                box_target[..., 2:],
                weight=box_weight,
                avg_factor=avg_factor,
            )
            loss_box = loss_xy + loss_wh
            loss_dict["loss_box_{}".format(scale)] = loss_box

            # cls loss
            loss_cls = self.loss_cls(
                pred_cls, cls_target, weight=conf_target, avg_factor=avg_factor
            )
            loss_dict["loss_cls_{}".format(scale)] = loss_cls

            # conf loss
            results = self.box_coder.decode(
                outputs[scale], anchors, downsample
            )
            pred_boxes = results["pred_boxes"]
            ious = self.iou_calculator(pred_boxes, gt_boxes)  # [B, N, M]
            ious = ious.max(axis=-1).reshape([B, H, W, -1, 1])
            ignore_mask = (1 - conf_target) * (ious > self.ignore_thresh)
            ignore_mask.stop_gradient = True
            loss_conf = self.loss_conf(
                pred_conf,
                conf_target,
                weight=(1 - ignore_mask),
                avg_factor=avg_factor,
            )
            loss_dict["loss_conf_{}".format(scale)] = loss_conf

            # iou loss
            if self.loss_iou is not None:
                loss_iou = self.loss_iou(
                    pred_boxes,
                    gt_boxes,
                    weight=box_weight,
                    avg_factor=avg_factor,
                )
                loss_dict["loss_iou_{}".format(scale)] = loss_iou

            # iou aware loss
            if self.iou_aware:
                loss_iou_aware = self.loss_iou_aware(
                    pred_iou, ious, weight=conf_target, avg_factor=avg_factor
                )
                loss_dict["loss_iou_aware_{}".format(scale)] = loss_iou_aware

        return loss_dict

    def predict(self, pred_dict: dict, **kwargs) -> list:
        """predict"""
        outputs = pred_dict["outputs"]
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        multi_scale_results = []
        for output, anchors, downsample in zip(
            outputs, self.anchors, self.downsamples
        ):
            results = self.box_coder.decode(
                output, anchors, downsample, filter=True
            )
            multi_scale_results.append(results)

        # merge multi-scale results
        results = []
        batch_size = outputs[0].shape[0]
        for batch_id in range(batch_size):
            sub_results = dict()
            for key in multi_scale_results[0][0]:
                num_boxes = sum(
                    [
                        len(single_scale_results[batch_id][key])
                        for single_scale_results in multi_scale_results
                    ]
                )
                if num_boxes:
                    sub_results[key] = paddle.concat(
                        [
                            single_scale_results[batch_id][key]
                            for single_scale_results in multi_scale_results
                            if len(single_scale_results[batch_id][key])
                        ],
                        axis=0,
                    )
                else:
                    sub_results[key] = multi_scale_results[0][batch_id][key]

            results.append(sub_results)

        # nms
        if self.nms is not None:
            results = self.nms(results)

        return results


if __name__ == "__main__":
    head = YOLOv3Head(
        in_channels=[1024, 512, 256],
        anchors=[
            [10, 13],
            [16, 30],
            [33, 23],
            [30, 61],
            [62, 45],
            [59, 119],
            [116, 90],
            [156, 198],
            [373, 326],
        ],
        anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    )
    feats = [
        paddle.randn([4, 256, 640, 640]),
        paddle.randn([4, 512, 320, 320]),
        paddle.randn([4, 1024, 160, 160]),
    ]
    outputs = head(feats)["outputs"]
    for output in outputs:
        print(output.shape)
