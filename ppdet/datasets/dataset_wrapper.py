# -*- encoding: utf-8 -*-
"""
@File    :   dataset_wrapper.py
@Time    :   2024/01/31 21:01:14
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import numpy as np
import logging
from ppdet.datasets import BaseDataset
from ppdet.apis import manager


@manager.DATASETS.add_component
class CBGSDataset(BaseDataset):
    """
    Class-balanced Grouping and Sampling Dataset
    """

    def __init__(
        self, 
        dataset,
        ignore_empty_sample=False,
    ):
        self.ignore_empty_sample = ignore_empty_sample
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.cat2id = dataset.cat2id
        self.sample_indices = self._get_sample_indices()


    def _get_sample_indices(self):
        """Get sample indices"""
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)).tolist()

        if not self.ignore_empty_sample:
            indices = list(range(len(self.dataset)))
            empty_indices = list(set(indices) - set(sample_indices))
            sample_indices += empty_indices

        return sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        idx = self.sample_indices[idx]
        return self.dataset[idx]

    @property
    def name(self):
        """name"""
        return self.dataset.name

    @property
    def is_train_mode(self):
        """is train mode"""
        return self.dataset.is_train_mode

    @property
    def is_test_mode(self):
        """is test mode"""
        return self.dataset.is_test_mode

    def collate_fn(self, batch):
        """collate_fn"""
        return self.dataset.collate_fn(batch)

    def set_epoch(self, epoch):
        """set epoch"""
        self.dataset.set_epoch(epoch)


if __name__ == "__main__":
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

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
    }
    from ppdet.datasets import BoschTrafficLightDataset
    dataset = BoschTrafficLightDataset(**dataset)
    print("Dataset: ", len(dataset))
    dataset = CBGSDataset(dataset)
    print("CBGSDataset: ", len(dataset))
