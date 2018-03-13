import tensorflow as tf
import numpy as np
from lib.networks.base_network import Net


class InceptionV1(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        self.x = tf.placeholder(tf.float32, name='x', shape=[self.config.batch_size,
                                                             self.config.image_width,
                                                             self.config.image_height,
                                                             self.config.image_depth], )
        self.y = tf.placeholder(tf.int16, name='y', shape=[self.config.batch_size,
                                                           self.config.n_classes])
        self.loss = None
        self.accuracy = None
        self.summary = []
        # self.layers = {}

    def init_saver(self):
        pass

    def conv2d(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='same'):
        in_channels = inputs.get_shape()[-1]
        with tf.variable_scope(layer_name):
            w = tf.get_variable(name='weights',
                                trainable=True,
                                shape=[kernel_size, kernel_size, in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=True,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, w, [1, strides, strides, 1], padding=padding, name='conv')
            inputs = tf.nn.bias_add(inputs, b, name='bias_add')
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs

    def max_pool(self, layer_name, inputs, pool_size, strides, padding='same'):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding, name=layer_name)

    def lrn(self, layer_name, inputs, depth_radius=5, alpha=0.0001, beta=0.75):
        with tf.name_scope(layer_name):
            return tf.nn.local_response_normalization(name='pool1_norm1', inputs=inputs, depth_radius=depth_radius, alpha=alpha, beta=beta)

    def build_model(self):
        # conv1_7x7_s2 = tf.layers.conv2d(name='conv1_7x7_s2', inputs=self.x,
        #                                 filters=64, kernel_size=7, strides=2, padding='same',
        #                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                                 bias_initializer=tf.constant_initializer(0.2),
        #                                 activation=tf.nn.relu)
        # pool1_3x3_s2 = tf.layers.max_pooling2d(name='pool1_3x3_s2', inputs=conv1_7x7_s2,
        #                                        pool_size=3, strides=2, padding='same')
        # pool1_norm1 = tf.nn.local_response_normalization(name='pool1_norm1', inputs=pool1_3x3_s2,
        #                                                  depth_radius=5, alpha=0.0001, beta=0.75)
        # conv2_3x3_reduce = tf.layers.conv2d(name='conv2_3x3_reduce', inputs=pool1_norm1,
        #                                     filters=64, kernel_size=1, padding='same',
        #                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                                     bias_initializer=tf.constant_initializer(0.2),
        #                                     activation=tf.nn.relu)
        conv1_7x7_s2 = self.conv2d('conv1_7x7_s2', self.x, 64, 7, 2)
        pool1_3x3_s2 = self.max_pool('pool1_3x3_s2', conv1_7x7_s2, 3, 2)
        pool1_norm1 = self.lrn('pool1_norm1', pool1_3x3_s2)
        conv2_3x3_reduce = self.conv2d('conv2_3x3_reduce', pool1_norm1, 64, 1, 1)
        conv2_3x3 = self.conv2d('conv2_3x3', conv2_3x3_reduce, 192, 3, 1)
        conv2_norm2 = self.lrn('conv2_norm2', conv2_3x3)
        pool2_3x3_s2 = self.max_pool('pool2_3x3_s2', conv2_norm2, 3, 2)
