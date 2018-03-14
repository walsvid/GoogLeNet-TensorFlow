import os
import tensorflow as tf
import numpy as np
from lib.utils.config import DataConfig


class DataLoader(object):
    def __init__(self, config, is_train, is_shuffle):
        """
        :param config: input config
        :type config: DataConfig
        :param is_train: is in train phase
        :type is_train: bool
        :param is_shuffle: shuffle data
        :type is_shuffle: bool
        """
        self.config = config
        self.is_train = is_train
        self.is_shuffle = is_shuffle

    def load_data(self):
        pass

    def generate_batch(self):
        pass


class Flowers102DataLoader(DataLoader):
    def __init__(self, config, is_train, is_shuffle):
        super().__init__(config, is_train, is_shuffle)
