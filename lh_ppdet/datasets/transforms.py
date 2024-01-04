# -*- encoding: utf-8 -*-
"""
@File    :   transforms.py
@Time    :   2023/11/26 15:19:21
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import cv2
import numpy as np
import random
import pickle
import logging
import xml.etree.ElementTree as ET
import paddle.vision.transforms as T
from lh_ppdet.apis import manager


__all__ = [
    "Compose",
    "LoadImageFromFile",
    "LoadLabelFromFile",
    "ColorJitter",
    "RandomVerticalFlip",
    "RandomHorizontalFlip",
    "ResizeImage",
    "RandomScaleImage",
    "RandomErasingImage",
    "RandomCropImage",
    "CopyAndPalse",
    "ShuffleLabel",
    "NormalizeImage",
    "Visualize",
]


class Compose:
    """
    Compose
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError("The transforms must be a list!")
        self.transforms = transforms

    def __call__(self, sample):
        """ """
        for t in self.transforms:
            sample = t(sample)

        return sample


@manager.TRANSFORMS.add_component
class LoadImageFromFile:
    """
    Load image from file
    """

    def __init__(self, to_float=False):
        self.to_float = to_float

    def __call__(self, sample):
        image_file = sample["img_meta"]["image_file"]
        image = cv2.imread(image_file)
        if self.to_float:
            image = image.astype("float32")

        sample["image"] = image
        sample["img_meta"].update(
            dict(
                ori_size=(image.shape[1], image.shape[0]),
                img_size=(image.shape[1], image.shape[0]),
            )
        )

        return sample


@manager.TRANSFORMS.add_component
class LoadLabelFromFile:
    """
    Load label from file
    """

    def __init__(self, class_names):
        self.class2id = dict(zip(class_names, range(len(class_names))))

    def __call__(self, sample):
        label_file = sample["img_meta"]["label_file"]
        if not label_file:
            return sample

        tree = ET.parse(label_file)
        objs = tree.findall("object")
        image_width = float(tree.find("size").find("width").text)
        image_height = float(tree.find("size").find("height").text)
        gt_boxes, gt_names, gt_labels = [], [], []
        for obj in objs:
            class_name = obj.find("name").text
            if class_name not in self.class2id:
                continue
            gt_label = self.class2id[class_name]
            x1 = float(obj.find("bndbox").find("xmin").text)
            y1 = float(obj.find("bndbox").find("ymin").text)
            x2 = float(obj.find("bndbox").find("xmax").text)
            y2 = float(obj.find("bndbox").find("ymax").text)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_width - 1, x2)
            y2 = min(image_height - 1, y2)
            gt_box = [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                x2 - x1 + 1,
                y2 - y1 + 1,
            ]
            gt_boxes.append(gt_box)
            gt_names.append(class_name)
            gt_labels.append(gt_label)
        if len(gt_boxes) == 0:  # add a fade box
            gt_boxes = np.zeros((0, 4), dtype="float32")
            gt_names = np.zeros((0,), dtype="str")
            gt_labels = np.zeros((0,), dtype="int32")

        sample["gt_boxes"] = np.array(gt_boxes, dtype="float32")
        sample["gt_names"] = np.array(gt_names, dtype="str")
        sample["gt_labels"] = np.array(gt_labels, dtype="int32")
        if "img_size" not in sample["img_meta"]:
            sample["img_meta"]["img_size"] = (image_width, image_height)

        return sample


@manager.TRANSFORMS.add_component
class ColorJitter:
    """
    ColorJitter
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])

        return sample


@manager.TRANSFORMS.add_component
class RandomVerticalFlip:
    """
    Random vertical flip
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample["image"] = sample["image"][::-1]
            gt_boxes = sample.get("gt_boxes", [])
            if len(gt_boxes):
                height = sample["image"].shape[0]
                sample["gt_boxes"][:, 1] = (
                    height - 1 - sample["gt_boxes"][:, 1]
                )

        return sample


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip:
    """
    Random horizontal flip
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample["image"] = sample["image"][:, ::-1]
            gt_boxes = sample.get("gt_boxes", [])
            if len(gt_boxes):
                width = sample["image"].shape[1]
                sample["gt_boxes"][:, 0] = width - 1 - sample["gt_boxes"][:, 0]

        return sample


@manager.TRANSFORMS.add_component
class ResizeImage:
    """
    Resize image
    """

    def __init__(self, size, interp=cv2.INTER_LINEAR):
        self.size = size
        self.interp = interp

    def __call__(self, sample):
        image = sample["image"]
        ori_size = (image.shape[1], image.shape[0])
        sample["image"] = cv2.resize(
            image, self.size, interpolation=self.interp
        )
        sample["img_meta"]["img_size"] = self.size
        gt_boxes = sample.get("gt_boxes", [])
        if len(gt_boxes):
            sx = self.size[0] / ori_size[0]
            sy = self.size[1] / ori_size[1]
            gt_boxes[:, 0] *= sx
            gt_boxes[:, 1] *= sy
            gt_boxes[:, 2] *= sx
            gt_boxes[:, 3] *= sy
            sample["gt_boxes"] = gt_boxes

        return sample


@manager.TRANSFORMS.add_component
class RandomScaleImage:
    """
    Random scale image
    """

    def __init__(self, scales, interp=cv2.INTER_LINEAR):
        assert len(scales) == 2, "len of scales should be 2"
        self.scales = scales
        self.interp = interp

    def __call__(self, sample):
        scale = (
            random.random() * (self.scales[1] - self.scales[0])
            + self.scales[0]
        )
        image = cv2.resize(
            sample["image"],
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=self.interp,
        )
        sample["image"] = image
        sample["img_meta"]["img_size"] = (image.shape[1], image.shape[0])
        gt_boxes = sample.get("gt_boxes", [])
        if len(gt_boxes):
            gt_boxes *= scale
            sample["gt_boxes"] = gt_boxes

        return sample


@manager.TRANSFORMS.add_component
class RandomErasingImage:
    """
    Random erasing image
    """

    def __init__(
        self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
    ):
        self.transform = T.RandomErasing(prob, scale, ratio, value)

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])

        return sample


@manager.TRANSFORMS.add_component
class RandomCropImage:
    """
    Random crop image
    """

    def __init__(self, size, prob=0.5):
        self.size = size
        self.prob = prob

    def __call__(self, sample):
        img_size = sample["img_meta"]["img_size"]
        assert (
            img_size[0] >= self.size[0] and img_size[1] >= self.size[1]
        ), "img_size should be larger than crop size"
        if random.random() < self.prob:
            offset_x = random.randint(0, img_size[0] - self.size[0])
            offset_y = random.randint(0, img_size[1] - self.size[1])
            sample["image"] = sample["image"][
                offset_y : offset_y + self.size[1],
                offset_x : offset_x + self.size[0],
            ]
            sample["img_meta"]["img_size"] = self.size

            gt_boxes = sample.get("gt_boxes", [])
            if len(gt_boxes):
                gt_boxes[:, 0] -= offset_x
                gt_boxes[:, 1] -= offset_y
                sample["gt_boxes"] = gt_boxes
                mask = np.logical_and(
                    np.logical_and(
                        gt_boxes[:, 0] >= 0, gt_boxes[:, 0] < self.size[0]
                    ),
                    np.logical_and(
                        gt_boxes[:, 1] >= 0, gt_boxes[:, 1] < self.size[1]
                    ),
                )
                sample["gt_boxes"] = sample["gt_boxes"][mask]
                sample["gt_names"] = sample["gt_names"][mask]
                sample["gt_labels"] = sample["gt_labels"][mask]

        return sample


@manager.TRANSFORMS.add_component
class CopyAndPalse:
    """
    Copy and paste
    """

    def __init__(
        self, class_names, db_sampler_file, num_samples=5, max_retry=5
    ):
        self.class_names = class_names
        self.class2id = dict(zip(class_names, range(len(class_names))))
        self.db_sampler_file = db_sampler_file
        if not isinstance(num_samples, dict):
            num_samples = {gt_name: num_samples for gt_name in class_names}
        self.num_samples = num_samples
        self.max_retry = max_retry

        with open(db_sampler_file, "rb") as f:
            self.gt_samples = pickle.load(f)

        logging.info("GT samples: ")
        for key, value in self.gt_samples.items():
            logging.info("{}: {}".format(key, len(value)))

    def __call__(self, sample):
        image = sample["image"]
        height, width = image.shape[:2]
        gt_boxes = sample["gt_boxes"]
        gt_names = sample["gt_names"]
        gt_labels = sample["gt_labels"]

        # mask
        mask = np.zeros_like(image[..., 0], dtype="bool")
        for gt_box in gt_boxes:
            x_min, y_min = np.floor(gt_box[:2] - gt_box[2:] / 2).astype("int")
            x_max, y_max = np.ceil(gt_box[:2] + gt_box[2:] / 2).astype("int")
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(height - 1, x_max)
            y_max = min(width - 1, y_max)
            mask[y_min:y_max, x_min:x_max] = True

        # sample from gt_samples
        sampled_gt_boxes, sampled_gt_names, sampled_gt_labels = [], [], []
        for gt_name, gt_samples in self.gt_samples.items():
            gt_label = self.class2id[gt_name]
            num = (gt_names == gt_name).sum()
            if gt_label == 5:
                a = 1
            if num < self.num_samples[gt_name]:
                num_sample = self.num_samples[gt_name] - num
                for _ in range(num_sample):
                    idx = random.randint(0, len(gt_samples) - 1)
                    gt_sample = gt_samples[idx]
                    gt_box = gt_sample["gt_boxes"].copy()
                    patch = gt_sample["image"]
                    patch_h, patch_w = patch.shape[:2]

                    for _ in range(self.max_retry):
                        offset_x = random.randint(0, width - patch_w - 1)
                        offset_y = random.randint(0, height - patch_h - 1)
                        pts = [
                            [offset_x, offset_y],
                            [offset_x + patch_w, offset_y],
                            [offset_x, offset_y + patch_h],
                            [offset_x + patch_w, offset_y + patch_h],
                        ]
                        valid = True
                        for pt in pts:
                            if mask[pt[1], pt[0]]:
                                valid = False
                                break
                        if valid:
                            break
                    if not valid:
                        continue

                    # add sample
                    gt_box[0] += offset_x
                    gt_box[1] += offset_y
                    sampled_gt_boxes.append(gt_box)
                    sampled_gt_names.append(gt_name)
                    sampled_gt_labels.append(gt_label)
                    image[
                        offset_y : offset_y + patch_h,
                        offset_x : offset_x + patch_w,
                    ] = patch
                    mask[
                        offset_y : offset_y + patch_h,
                        offset_x : offset_x + patch_w,
                    ] = True

        sampled_gt_boxes = np.array(sampled_gt_boxes, dtype="float32").reshape(
            [-1, 4]
        )
        sampled_gt_names = np.array(sampled_gt_names, dtype="str")
        sampled_gt_labels = np.array(sampled_gt_labels, dtype="int32")

        sample["gt_boxes"] = np.concatenate([gt_boxes, sampled_gt_boxes])
        sample["gt_names"] = np.concatenate([gt_names, sampled_gt_names])
        sample["gt_labels"] = np.concatenate([gt_labels, sampled_gt_labels])

        return sample


@manager.TRANSFORMS.add_component
class NormalizeImage:
    """
    Normalize image
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"].astype("float32")
        image = (image - self.mean) / self.std
        sample["image"] = image.transpose((2, 0, 1)).astype("float32")

        return sample


@manager.TRANSFORMS.add_component
class ShuffleLabel:
    """
    Shuffle label
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        gt_boxes = sample.get("gt_boxes", [])
        if len(gt_boxes):
            indices = np.arange(len(gt_boxes))
            np.random.shuffle(indices)
            sample["gt_boxes"] = sample["gt_boxes"][indices]
            sample["gt_names"] = sample["gt_names"][indices]
            sample["gt_labels"] = sample["gt_labels"][indices]

        return sample


@manager.TRANSFORMS.add_component
class Visualize:
    """
    Visualize
    """

    def __init__(self, save_path, with_label=False):
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.with_label = with_label

    def __call__(self, sample):
        image = sample["image"].copy()
        if self.with_label and "gt_boxes" in sample:
            gt_boxes = sample["gt_boxes"]
            gt_labels = sample["gt_labels"]
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                if gt_label < 0:
                    continue
                pt = (int(gt_box[0]), int(gt_box[1]))
                pt1 = (
                    int(gt_box[0] - gt_box[2] / 2),
                    int(gt_box[1] - gt_box[3] / 2),
                )
                pt2 = (
                    int(gt_box[0] + gt_box[2] / 2),
                    int(gt_box[1] + gt_box[3] / 2),
                )
                cv2.rectangle(image, pt1, pt2, (0, 0, 255))
                cv2.putText(
                    image,
                    str(gt_label),
                    pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                )

        filename = os.path.basename(sample["img_meta"]["image_file"])
        save_file = os.path.join(self.save_path, filename)
        cv2.imwrite(save_file, image)

        return sample
