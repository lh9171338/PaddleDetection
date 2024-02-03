# -*- encoding: utf-8 -*-
"""
@File    :   loss.py
@Time    :   2023/12/18 19:28:38
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn.functional as F
from ppdet.apis import manager
from ppdet.losses.utils import weighted_loss


__all__ = [
    "BCELoss",
    "CELoss",
    "L1Loss",
    "FocalLoss",
]


class Loss:
    """
    Loss
    """

    def __init__(
        self,
        weight=None,
        **kwargs,
    ):
        if isinstance(weight, (list, tuple)):
            weight = paddle.to_tensor(weight)

        self.weight = weight

    def __call__(self, pred, target):
        raise NotImplementedError


@manager.LOSSES.add_component
class BCELoss(Loss):
    """
    Binary Coss Entropy Loss
    """

    def __init__(
        self,
        with_logits=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.with_logits = with_logits

    @weighted_loss
    def __call__(self, pred, target):
        if self.with_logits:
            loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction="none"
            )
        else:
            loss = F.binary_cross_entropy(pred, target, reduction="none")

        if self.weight is not None:
            loss = loss * self.weight

        return loss


@manager.LOSSES.add_component
class CELoss(Loss):
    """
    Coss Entropy Loss
    """

    def __init__(
        self,
        with_logits=False,
        soft_label=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.with_logits = with_logits
        self.soft_label = soft_label

    @weighted_loss
    def __call__(self, pred, target):
        if self.with_logits:
            loss = F.softmax_with_cross_entropy(
                pred, target, soft_label=self.soft_label
            )
        else:
            loss = F.cross_entropy(
                pred, target, soft_label=self.soft_label, reduction="none"
            )

        if self.weight is not None:
            loss = loss * self.weight

        return loss


@manager.LOSSES.add_component
class L1Loss(Loss):
    """
    L1 Loss
    """

    @weighted_loss
    def __call__(self, pred, target):
        loss = paddle.abs(pred - target)
        if self.weight is not None:
            loss = loss * self.weight

        return loss


@manager.LOSSES.add_component
class FocalLoss(Loss):
    """A wrapper around paddle.nn.functional.sigmoid_focal_loss.
    Args:
        use_sigmoid (bool): currently only support use_sigmoid=True
        alpha (float): parameter alpha in Focal Loss
        gamma (float): parameter gamma in Focal Loss
        loss_weight (float): final loss will be multiplied by this
    """
    def __init__(
        self,
        use_sigmoid=True,
        alpha=0.25,
        gamma=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert use_sigmoid == True, \
            'Focal Loss only supports sigmoid at the moment'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma

    @weighted_loss
    def __call__(self, pred, target):
        """forward function.
        Args:
            pred (Tensor): logits of class prediction, of shape (N, num_classes)
            target (Tensor): target class label, of shape (N, )
        """
        loss = F.sigmoid_focal_loss(
            pred, target, alpha=self.alpha, gamma=self.gamma, reduction="none")        
        if self.weight is not None:
            loss = loss * self.weight

        return loss
