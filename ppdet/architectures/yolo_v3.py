# -*- encoding: utf-8 -*-
"""
@File    :   yolo_detector.py
@Time    :   2023/12/16 23:10:44
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


from ppdet.architectures import Detector
from ppdet.apis import manager


@manager.MODELS.add_component
class YOLOv3(Detector):
    """
    YOLO v3 Detector
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_train(self, image, gt_boxes, gt_labels, **kwargs) -> dict:
        """foward function for training"""
        feats = self.backbone(image)
        if self.with_neck:
            feats = self.neck(feats)
        pred_dict = self.head(feats)
        loss_dict = self.head.loss(pred_dict, gt_boxes, gt_labels)

        return loss_dict

    def forward_test(self, image, **kwargs) -> list:
        """foward function for testing"""
        feats = self.backbone(image)
        if self.with_neck:
            feats = self.neck(feats)
        pred_dict = self.head(feats)
        results = self.head.predict(pred_dict)

        return results
