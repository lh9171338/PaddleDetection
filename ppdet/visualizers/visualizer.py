# -*- encoding: utf-8 -*-
"""
@File    :   visualizer.py
@Time    :   2023/12/21 22:46:13
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import cv2
import numpy as np
from ppdet.apis import manager
import ppdet.apis.utils as api_utils
from lh_tool.Iterator import SingleProcess, MultiProcess


__all__ = ["Visualizer", "BoxVisualizer"]


class Visualizer:
    """
    Visualizer
    """

    def __init__(
        self,
        class_names,
        black_list=[],
        score_thresh=0,
        nprocs=1,
        **kwargs,
    ):
        self.class_names = class_names
        self.black_list = black_list
        self.score_thresh = score_thresh
        self.nprocs = nprocs

        num_classes = len(class_names)
        colors = np.ones((1, num_classes, 3), dtype="float32")
        colors[:, :, 0] = np.linspace(0, 360, num_classes)
        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2RGB)
        colors = (colors * 255).astype("uint8")
        colors = [color.tolist() for color in colors[0]]
        self.color_dict = dict(zip(class_names, colors))

        self.reset()

    def reset(self):
        """reset"""
        self.result_buffer = []

    def update(self, results):
        """
        update

        Args:
            results (dict|list[dict]): prediction and target
        """
        if not isinstance(results, list):
            results = [results]
        results = api_utils.tensor2numpy(results)
        self.result_buffer.extend(results)

    def visualize(self, save_dir):
        """
        visualize

        Args:
            save_dir (str): save directory

        Returns:
            None
        """
        raise NotImplementedError


@manager.VISUALIZERS.add_component
class BoxVisualizer(Visualizer):
    """
    Box Visualizer
    """

    def _draw_box(
        self,
        image,
        boxes,
        fill=False,
        alpha=0.4,
        plot_text=False,
        box_color=None,
        text_color=None,
    ):
        """
        draw boxes

        Args:
            image (np.ndarray): image, shape [H, W, 3]
            boxes (np.ndarray): boxes, shape [N, 6], in order of x, y, w, h, score, class
            fill (bool): whether to fill box
            alpha (float): alpha
            plot_text (bool): whether to plot text
            box_color (list): box color
            text_color (list): text color

        Returns:
            None
        """
        if fill:
            fg_image = image.copy()
        for box in boxes:
            score = box[-2]
            class_name = self.class_names[int(box[-1])]
            pt = (int(box[0]), int(box[1]))
            pt1 = (
                int(box[0] - box[2] / 2),
                int(box[1] - box[3] / 2),
            )
            pt2 = (
                int(box[0] + box[2] / 2),
                int(box[1] + box[3] / 2),
            )
            color = (
                self.color_dict[class_name] if box_color is None else box_color
            )
            if fill:
                cv2.rectangle(fg_image, pt1, pt2, color, -1)
            else:
                cv2.rectangle(image, pt1, pt2, color)
            if plot_text:
                color = (
                    self.color_dict[class_name]
                    if text_color is None
                    else text_color
                )
                cv2.putText(
                    image,
                    "{:.2f}".format(score),
                    pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                )

        if fill:
            cv2.addWeighted(fg_image, alpha, image, 1 - alpha, 0, image)

    def _draw_legend(self, image):
        """
        draw legend

        Args:
            image (np.ndarray): image, shape [H, W, 3]

        Returns:
            None
        """
        margin = 10
        for i, class_name in enumerate(self.class_names):
            color = self.color_dict[class_name]
            pt1 = (margin, margin * (i * 2 + 1))
            pt2 = (margin * 2, margin * (i * 2 + 2))
            pt = (margin * 2, margin + margin * (i * 2 + 1))
            cv2.rectangle(image, pt1, pt2, color)
            cv2.putText(
                image, class_name, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color
            )

    def _visualize_single(self, save_dir, result):
        """
        visualize single image

        Args:
            save_dir (str): save directory
            result (dict): prediction and target

        Returns:
            None
        """
        # read image
        img_size = result["img_size"]
        ori_size = result["ori_size"]
        sx = ori_size[0] / img_size[0]
        sy = ori_size[1] / img_size[1]
        image_file = result["image_file"]
        image = cv2.imread(image_file)

        # target
        if "gt_boxes" in result:
            gt_boxes = result["gt_boxes"]
            gt_labels = result["gt_labels"]
            gt_boxes[:, [0, 2]] *= sx
            gt_boxes[:, [1, 3]] *= sy
            targets = []
            for box, label in zip(gt_boxes, gt_labels):
                if label < 0:  # filter invalid gt
                    continue
                class_name = self.class_names[label]
                if class_name in self.black_list:
                    continue
                target = box.tolist() + [1, label]
                targets.append(target)
            targets = np.array(targets)
            self._draw_box(image, targets, fill=True)

        # pred
        if "pred_boxes" in result:
            pred_boxes = result["pred_boxes"]
            pred_scores = result["pred_scores"]
            pred_labels = result["pred_labels"]
            pred_boxes[:, [0, 2]] *= sx
            pred_boxes[:, [1, 3]] *= sy
            preds = []
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score < self.score_thresh:
                    continue
                class_name = self.class_names[label]
                if class_name in self.black_list:
                    continue
                pred = box.tolist() + [score, label]
                preds.append(pred)
            preds = np.array(preds)
            self._draw_box(image, preds, plot_text=True)

        self._draw_legend(image)

        # save
        save_file = os.path.join(save_dir, os.path.basename(image_file))
        cv2.imwrite(save_file, image)

    def visualize(self, save_dir):
        """
        visualize

        Args:
            save_dir (str): save directory

        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)
        process = MultiProcess if self.nprocs > 1 else SingleProcess
        process(self._visualize_single, nprocs=self.nprocs).run(
            save_dir=save_dir,
            result=self.result_buffer,
        )
