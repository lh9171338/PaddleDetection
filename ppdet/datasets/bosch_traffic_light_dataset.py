# -*- encoding: utf-8 -*-
"""
@File    :   bosch_traffic_light_dataset.py
@Time    :   2023/11/26 15:12:46
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import pickle
from collections import defaultdict
import logging
import cv2
import numpy as np
import paddle
import tqdm
from ppdet.datasets import BaseDataset
from ppdet.apis import manager


@manager.DATASETS.add_component
class BoschTrafficLightDataset(BaseDataset):
    """
    BoschTrafficLightDataset
    """

    def __init__(
        self,
        data_root,
        ann_file,
        mode,
        class_names,
        pipeline=None,
        ignore_empty_sample=False,
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            mode=mode,
            class_names=class_names,
            pipeline=pipeline,
        )

        self.ignore_empty_sample = ignore_empty_sample
        if ignore_empty_sample:
            self.data_infos = [info for info in self.data_infos if "gt_boxes" not in info or len(info["gt_boxes"]) > 0]

        logging.info("len of {} dataset is: {}".format(self.mode, len(self)))

    def __getitem__(self, index):
        info = self.data_infos[index]
        image_file = os.path.join(
            self.data_root, info["img_meta"]["image_file"]
        )

        sample = {
            "mode": self.mode,
            "img_meta": {
                "sample_idx": index,
                "epoch": self.epoch,
                "image_file": image_file,
            },
        }
        if not self.is_test_mode:
            gt_boxes, gt_names, gt_labels = [], [], []
            for gt_box, gt_name in zip(info["gt_boxes"], info["gt_names"]):
                if gt_name in self.cat2id:
                    gt_label = self.cat2id[gt_name]
                    gt_boxes.append(gt_box)
                    gt_names.append(gt_name)
                    gt_labels.append(gt_label)

            sample["gt_boxes"] = np.array(gt_boxes, dtype="float32").reshape(
                [-1, 4]
            )
            sample["gt_names"] = np.array(gt_names, dtype="str")
            sample["gt_labels"] = np.array(gt_labels, dtype="int32")

        if self.pipeline:
            sample = self.pipeline(sample)

        return sample

    def get_cat_ids(self, idx):
        """get category ids"""
        info = self.data_infos[idx]
        gt_labels = [self.cat2id[gt_name] for gt_name in info["gt_names"] if gt_name in self.cat2id]
        gt_labels = set(gt_labels)

        return gt_labels

    def collate_fn(self, batch):
        """collate_fn"""
        sample = batch[0]
        collated_batch = {}
        for key in sample:
            if key in ["image"]:
                collated_batch[key] = np.stack(
                    [elem[key] for elem in batch], axis=0
                )
            elif key in ["gt_boxes"]:
                # padding 0
                max_num = max(max([len(elem[key]) for elem in batch]), 1)
                elems = []
                for elem in batch:
                    elem = elem[key]
                    elem = np.concatenate(
                        [elem, np.zeros((max_num - len(elem), 4))]
                    )
                    elems.append(elem)
                collated_batch[key] = np.stack(elems, axis=0)
            elif key in ["gt_labels"]:
                # padding -1
                max_num = max(max([len(elem[key]) for elem in batch]), 1)
                elems = []
                for elem in batch:
                    elem = elem[key]
                    elem = np.concatenate(
                        [elem, np.full((max_num - len(elem),), -1)]
                    )
                    elems.append(elem)
                collated_batch[key] = np.stack(elems, axis=0)
            elif key in ["img_meta"]:
                collated_batch[key] = [elem[key] for elem in batch]

        return collated_batch

    def generate_anchors(self, num_anchors=9, scaled_img_size=None):
        """generate anchors"""
        box_sizes = []
        for sample in self.data_infos:
            gt_boxes = sample["gt_boxes"]
            img_size = sample["img_meta"]["img_size"]
            if scaled_img_size is not None and len(gt_boxes):
                gt_boxes[:, 2] *= scaled_img_size[0] / img_size[0]
                gt_boxes[:, 3] *= scaled_img_size[1] / img_size[1]
            box_sizes.append(gt_boxes[:, 2:])
        box_sizes = np.vstack(box_sizes)

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_anchors).fit(box_sizes)
        anchors = np.array(kmeans.cluster_centers_)
        anchors.sort(axis=0)

        return anchors

    def generate_db_sampler(self, save_file, expand=2):
        """generate db sampler"""
        gt_samples = defaultdict(list)
        for info in tqdm.tqdm(self.data_infos):
            gt_boxes = info["gt_boxes"]
            gt_names = info["gt_names"]
            image_file = os.path.join(
                self.data_root, info["img_meta"]["image_file"]
            )
            image = cv2.imread(image_file)
            height, width = image.shape[:2]
            for gt_box, gt_name in zip(gt_boxes, gt_names):
                x_min, y_min = gt_box[:2] - gt_box[2:] / 2
                x_max, y_max = gt_box[:2] + gt_box[2:] / 2
                x_min = max(0, int(x_min) - expand)
                y_min = max(0, int(y_min) - expand)
                x_max = min(width - 1, int(x_max) + expand)
                y_max = min(height - 1, int(y_max) + expand)
                patch = image[y_min:y_max, x_min:x_max]
                gt_box[0] -= x_min
                gt_box[1] -= y_min
                gt_sample = {
                    "gt_boxes": gt_box,
                    "image": patch,
                }
                gt_samples[gt_name].append(gt_sample)

        logging.info("GT samples: ")
        for key, value in gt_samples.items():
            logging.info("{}: {}".format(key, len(value)))

        with open(save_file, "wb") as f:
            pickle.dump(gt_samples, f)


if __name__ == "__main__":
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    from ppdet.datasets.transforms import (
        LoadImageFromFile,
        LoadLabelFromFile,
        ColorJitter,
        RandomHorizontalFlip,
        ResizeImage,
        RandomScaleImage,
        RandomCropImage,
        ShuffleLabel,
        CopyAndPalse,
        Visualize,
        NormalizeImage,
    )

    class_names = [
        "RedLeft",
        "Red",
        "RedRight",
        "GreenLeft",
        "Green",
        "GreenRight",
        "Yellow",
        "off",
    ]
    dataset = {
        "type": "BoschTrafficLightDataset",
        "data_root": "data/bosch_traffic_light",
        "ann_file": "data/bosch_traffic_light/train-2832.pkl",
        "mode": "train",
        "class_names": class_names,
        "pipeline": [
            LoadImageFromFile(),
            CopyAndPalse(
                class_names=class_names,
                db_sampler_file="data/bosch_traffic_light/dbsampler.pkl",
                num_samples=5,
            ),
            # ColorJitter(0.4, 0.4, 0.4, 0.4),
            # RandomHorizontalFlip(0.5),
            # RandomCropImage((640, 640), 0.5),
            # ResizeImage(size=(640, 640)),
            # RandomScaleImage([0.5, 2]),
            Visualize(save_path="visualize", with_label=True),
            NormalizeImage(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
            ),
        ],
    }
    dataset = BoschTrafficLightDataset(**dataset)
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=4,
        num_workers=16,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    # dataset.generate_db_sampler("data/bosch_traffic_light/db_sampler.pkl")
    for sample in dataset:
        print(sample["img_meta"]["sample_idx"])
