# -*- encoding: utf-8 -*-
"""
@File    :   metric.py
@Time    :   2023/12/21 22:32:32
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import tqdm
from paddle.metric import Metric
from lh_ppdet.apis import manager
from lh_ppdet.boxes import IouSimilarity
import lh_ppdet.apis.utils as api_utils


__all__ = ["MAPMetric"]


def plot_pr_curve(save_file, prs, rcs, title="PR Curve", label=None):
    """
    Plot precision-recall curve

    Args:
        save_file (str): save file path
        prs (list): precision list
        rcs (list): recall list
        title (str): title
        legend (list): legend

    Return:
        None
    """
    plt.figure()
    plt.axis("equal")
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.plot(rcs, prs, "r-", label=label)
    plt.rc("legend", fontsize=10)
    plt.legend()

    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color=[0, 0.5, 0], alpha=0.3)
        plt.annotate(
            "f={0:0.1}".format(f_score),
            xy=(0.9, y[45] + 0.02),
            alpha=0.4,
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_file)
    plt.savefig(os.path.splitext(save_file)[0] + ".pdf")
    plt.close()


@manager.METRICS.add_component
class MAPMetric(Metric):
    """
    MAP Metric
    """

    def __init__(
        self,
        class_names,
        iou_thresh=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_thresh = iou_thresh

        self.iou_calculator = IouSimilarity()

        self.reset()

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__name__

    def reset(self):
        """reset"""
        self.tpfp_buffer = [[] for _ in range(self.num_classes)]
        self.pred_score_buffer = [[] for _ in range(self.num_classes)]
        self.gt_count_buffer = [0] * self.num_classes

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        if not isinstance(results, list):
            results = [results]

        for result in tqdm.tqdm(results):
            pred_boxes = result["pred_boxes"]
            pred_scores = result["pred_scores"]
            pred_labels = result["pred_labels"]
            gt_boxes = result["gt_boxes"].astype(pred_boxes.dtype)
            gt_labels = result["gt_labels"]

            # for each class
            for class_id in range(self.num_classes):
                # record gt count
                mask = gt_labels == class_id
                num_gt = mask.sum()
                if num_gt:
                    cur_gt_boxes = gt_boxes[mask].reshape([-1, 4])
                self.gt_count_buffer[class_id] += num_gt.item()

                # record pred score
                mask = pred_labels == class_id
                num_pred = mask.sum()
                if num_pred == 0:
                    continue
                cur_pred_boxes = pred_boxes[mask].reshape([-1, 4])
                cur_pred_scores = pred_scores[mask]
                self.pred_score_buffer[class_id].append(
                    cur_pred_scores.numpy()
                )

                # calculate iou
                tpfp = paddle.zeros((num_pred))
                if num_gt:
                    ious = self.iou_calculator(cur_pred_boxes, cur_gt_boxes)
                    gt_indices = paddle.argmax(ious, axis=-1)
                    visited = paddle.zeros((num_gt), dtype="bool")
                    for pred_idx, gt_idx in enumerate(gt_indices):
                        iou = ious[pred_idx, gt_idx]
                        if iou >= self.iou_thresh and not visited[gt_idx]:
                            tpfp[pred_idx] = 1
                            visited[gt_idx] = True

                self.tpfp_buffer[class_id].append(tpfp.numpy())

    def accumulate(self, save_dir=None) -> dict:
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            ap_dict (dict): ap dict
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # collect results from all ranks
        buffer = dict(
            gt_count_buffer=self.gt_count_buffer,
            tpfp_buffer=self.tpfp_buffer,
            pred_score_buffer=self.pred_score_buffer,
        )
        ret_list = api_utils.collect_object(buffer)
        self.reset()
        for ret in ret_list:
            for class_id in range(self.num_classes):
                self.gt_count_buffer[class_id] += ret["gt_count_buffer"][
                    class_id
                ]
                self.tpfp_buffer[class_id].extend(ret["tpfp_buffer"][class_id])
                self.pred_score_buffer[class_id].extend(
                    ret["pred_score_buffer"][class_id]
                )

        metrics = dict()
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            gt_count = self.gt_count_buffer[class_id]
            if gt_count == 0:
                continue
            if len(self.tpfp_buffer[class_id]) == 0:
                AP, P, R = 0, 0, 0
            else:
                tpfp = np.concatenate(self.tpfp_buffer[class_id])
                pred_score = np.concatenate(self.pred_score_buffer[class_id])

                # sort
                indices = np.argsort(-pred_score, axis=0)
                tpfp = tpfp[indices]

                tp = np.cumsum(tpfp, axis=0)
                fp = np.cumsum(1 - tpfp, axis=0)
                prs = tp / np.maximum(tp + fp, 1e-9)
                rcs = tp / gt_count
                AP, P, R = self._calc_AP(prs, rcs)

                if save_dir is not None:
                    save_file = os.path.join(
                        save_dir, "{}.jpg".format(class_name)
                    )
                    plot_pr_curve(
                        save_file, prs, rcs, label="AP={:.1f}".format(AP)
                    )

            metrics.update(
                {
                    "{}_AP".format(class_name): AP,
                    "{}_precision".format(class_name): P,
                    "{}_recall".format(class_name): R,
                }
            )

        metrics.update(
            {
                key: np.mean(
                    [value for name, value in metrics.items() if key in name]
                )
                for key in ["AP", "precision", "recall"]
            }
        )
        keys = ["mAP", "precision", "recall"] + [
            key.split("_AP")[0] for key, _ in metrics.items() if "_AP" in key
        ]
        values = [metrics["AP"], metrics["precision"], metrics["recall"]] + [
            value for key, value in metrics.items() if "_AP" in key
        ]
        values = ["{:.2f}".format(value * 100) for value in values]
        print("| " + " | ".join(keys) + " |")
        print("|" + " :---: |" * (len(keys) + 2))
        print("| " + " | ".join(values) + " |")

        return metrics

    def _calc_AP(self, prs, rcs):
        """
        calculate AP

        Args:
            prs (np.array): precision array
            rcs (np.array): recall array
        Return:
            AP (float)
            P (float)
            R (float)
        """
        precision = np.concatenate(([0.0], prs))
        recall = np.concatenate(([0.0], rcs))

        i = np.where(recall[1:] != recall[:-1])[0]
        AP = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        P, R = prs[-1], rcs[-1]

        return AP, P, R
