# -*- encoding: utf-8 -*-
"""
@File    :   base.py
@Time    :   2024/01/31 22:44:24
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import pickle
import paddle
from ppdet.datasets.transforms import Compose


class BaseDataset(paddle.io.Dataset):
    """
    Base Dataset
    """

    def __init__(
        self,
        data_root,
        ann_file,
        mode,
        class_names,
        pipeline=None,
        **kwargs,
    ):
        super().__init__()

        self.data_root = data_root
        self.ann_file = ann_file
        self.mode = mode
        self.class_names = class_names
        self.cat2id = dict(zip(class_names, range(len(class_names))))
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        self.epoch = 0

        self.load_annotations()

    @property
    def name(self):
        """name"""
        return self.__class__.__name__

    @property
    def is_train_mode(self):
        """is train mode"""
        return self.mode == "train"

    @property
    def is_test_mode(self):
        """is test mode"""
        return self.mode == "test"

    def set_epoch(self, epoch):
        """set epoch"""
        self.epoch = epoch

    def load_annotations(self):
        """load annotations"""
        with open(self.ann_file, "rb") as f:
            self.data_infos = pickle.load(f)

    def __len__(self):
        return len(self.data_infos)
