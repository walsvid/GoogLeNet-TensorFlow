import os
import tensorflow as tf
import numpy as np
import pandas as pd
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
        raise NotImplementedError

    def generate_batch(self):
        raise NotImplementedError


class Flowers102DataLoader(DataLoader):
    def __init__(self, config, is_train, is_shuffle):
        super().__init__(config, is_train, is_shuffle)
        self.image_width = self.config.image_width
        self.image_height = self.config.image_height
        self.image_depth = self.config.image_depth

        self.data_dir = self.config.data_dir
        self.batch_size = self.config.batch_size
        self.n_classes = self.config.n_classes

        self.input_queue = self.load_data()

    def load_data(self):
        image_dir = 'jpg'

        csv = pd.read_csv(os.path.join(self.config.data_dir, 'imagelabels.csv'), delimiter=',')
        image_list = csv['name'].map(lambda x: os.path.join(self.config.data_dir, image_dir, x)).values
        label_list = csv['labels'].values
        temp = np.array([image_list, label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(i) for i in label_list]

        image_list = tf.cast(image_list, tf.string)
        label_list = tf.cast(label_list, tf.int32)

        return tf.train.slice_input_producer([image_list, label_list])

    def generate_batch(self):
        label = self.input_queue[1]
        image_contents = tf.read_file(self.input_queue[0])
        image = tf.image.decode_jpeg(image_contents, channels=self.image_depth)

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [self.image_width, self.image_height])
        # image = tf.image.per_image_standardization(image)   # comment is when test get file and plot

        if self.is_shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=self.batch_size,
                                                              num_threads=64,
                                                              capacity=20000,
                                                              min_after_dequeue=3000)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size=self.batch_size,
                                                      num_threads=64,
                                                      capacity=20000)

        label_batch = tf.one_hot(label_batch, depth=self.n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [self.batch_size, self.n_classes])
        return image_batch, label_batch
