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

    def init_saver(self):
        pass

    def get_summary(self):
        return self.summary

    def conv2d(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME'):
        in_channels = inputs.get_shape()[-1]
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
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

    def max_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)

    def avg_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)

    def lrn(self, layer_name, inputs, depth_radius=5, alpha=0.0001, beta=0.75):
        with tf.name_scope(layer_name):
            return tf.nn.local_response_normalization(name='pool1_norm1', input=inputs, depth_radius=depth_radius,
                                                      alpha=alpha, beta=beta)

    def concat(self, layer_name, inputs):
        with tf.name_scope(layer_name):
            one_by_one = inputs[0]
            three_by_three = inputs[1]
            five_by_five = inputs[2]
            pooling = inputs[3]
            return tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)

    def dropout(self, layer_name, inputs, keep_prob):
        # dropout_rate = 1 - keep_prob
        with tf.name_scope(layer_name):
            return tf.nn.dropout(name=layer_name, x=inputs, keep_prob=keep_prob)

    def bn(self, layer_name, inputs, epsilon=1e-3):
        with tf.name_scope(layer_name):
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            inputs = tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_var, offset=None,
                                               scale=None, variance_epsilon=epsilon)
            return inputs

    def fc(self, layer_name, inputs, out_nodes):
        shape = inputs.get_shape()
        if len(shape) == 4:  # x is 4D tensor
            size = shape[1].value * shape[2].value * shape[3].value
        else:  # x has already flattened
            size = shape[-1].value
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(inputs, [-1, size])
            inputs = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            inputs = tf.nn.relu(inputs)
            return inputs

    def cal_loss(self, logits, labels):
        with tf.name_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross-entropy')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            loss_summary = tf.summary.scalar(scope, self.loss)
            self.summary.append(loss_summary)

    def cal_accuracy(self, logits, labels):
        with tf.name_scope('accuracy') as scope:
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct = tf.cast(correct, tf.float32)
            self.accuracy = tf.reduce_mean(correct) * 100.0
            accuracy_summary = tf.summary.scalar(scope, self.accuracy)
            self.summary.append(accuracy_summary)

    def optimize(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            return train_op

    def build_model(self):
        conv1_7x7_s2 = self.conv2d('conv1_7x7_s2', self.x, 64, 7, 2)
        pool1_3x3_s2 = self.max_pool('pool1_3x3_s2', conv1_7x7_s2, 3, 2)
        pool1_norm1 = self.lrn('pool1_norm1', pool1_3x3_s2)
        conv2_3x3_reduce = self.conv2d('conv2_3x3_reduce', pool1_norm1, 64, 1, 1)
        conv2_3x3 = self.conv2d('conv2_3x3', conv2_3x3_reduce, 192, 3, 1)
        conv2_norm2 = self.lrn('conv2_norm2', conv2_3x3)
        pool2_3x3_s2 = self.max_pool('pool2_3x3_s2', conv2_norm2, 3, 2)

        inception_3a_1x1 = self.conv2d('inception_3a_1x1', pool2_3x3_s2, 64, 1, 1)
        inception_3a_3x3_reduce = self.conv2d('inception_3a_3x3_reduce', pool2_3x3_s2, 96, 1, 1)
        inception_3a_3x3 = self.conv2d('inception_3a_3x3', inception_3a_3x3_reduce, 128, 3, 1)
        inception_3a_5x5_reduce = self.conv2d('inception_3a_5x5_reduce', pool2_3x3_s2, 16, 1, 1)
        inception_3a_5x5 = self.conv2d('inception_3a_5x5', inception_3a_5x5_reduce, 32, 5, 1)
        inception_3a_pool = self.max_pool('inception_3a_pool', pool2_3x3_s2, 3, 1)
        inception_3a_pool_proj = self.conv2d('inception_3a_pool_proj', inception_3a_pool, 32, 1, 1)
        inception_3a_output = self.concat('inception_3a_output', [inception_3a_1x1, inception_3a_3x3, inception_3a_5x5,
                                                                  inception_3a_pool_proj])

        inception_3b_1x1 = self.conv2d('inception_3b_1x1', inception_3a_output, 128, 1, 1)
        inception_3b_3x3_reduce = self.conv2d('inception_3b_3x3_reduce', inception_3a_output, 128, 1, 1)
        inception_3b_3x3 = self.conv2d('inception_3b_3x3', inception_3b_3x3_reduce, 192, 3, 1)
        inception_3b_5x5_reduce = self.conv2d('inception_3b_5x5_reduce', inception_3a_output, 32, 1, 1)
        inception_3b_5x5 = self.conv2d('inception_3b_5x5', inception_3b_5x5_reduce, 96, 5, 1)
        inception_3b_pool = self.max_pool('inception_3b_pool', inception_3a_output, 3, 1)
        inception_3b_pool_proj = self.conv2d('inception_3b_pool_proj', inception_3b_pool, 64, 1, 1)
        inception_3b_output = self.concat('inception_3b_output', [inception_3b_1x1, inception_3b_3x3, inception_3b_5x5,
                                                                  inception_3b_pool_proj])

        pool3_3x3_s2 = self.max_pool('pool3_3x3_s2', inception_3b_output, 3, 2)
        inception_4a_1x1 = self.conv2d('inception_4a_1x1', pool3_3x3_s2, 192, 1, 1)
        inception_4a_3x3_reduce = self.conv2d('inception_4a_3x3_reduce', pool3_3x3_s2, 96, 1, 1)
        inception_4a_3x3 = self.conv2d('inception_4a_3x3', inception_4a_3x3_reduce, 208, 3, 1)
        inception_4a_5x5_reduce = self.conv2d('inception_4a_5x5_reduce', pool3_3x3_s2, 16, 1, 1)
        inception_4a_5x5 = self.conv2d('inception_4a_5x5', inception_4a_5x5_reduce, 48, 5, 1)
        inception_4a_pool = self.max_pool('inception_4a_pool', pool3_3x3_s2, 3, 1)
        inception_4a_pool_proj = self.conv2d('inception_4a_pool_proj', inception_4a_pool, 64, 1, 1)
        inception_4a_output = self.concat('inception_4a_output', [inception_4a_1x1, inception_4a_3x3, inception_4a_5x5,
                                                                  inception_4a_pool_proj])

        inception_4b_1x1 = self.conv2d('inception_4b_1x1', inception_4a_output, 160, 1, 1)
        inception_4b_3x3_reduce = self.conv2d('inception_4b_3x3_reduce', inception_4a_output, 112, 1, 1)
        inception_4b_3x3 = self.conv2d('inception_4b_3x3', inception_4b_3x3_reduce, 224, 3, 1)
        inception_4b_5x5_reduce = self.conv2d('inception_4b_5x5_reduce', inception_4a_output, 24, 1, 1)
        inception_4b_5x5 = self.conv2d('inception_4b_5x5', inception_4b_5x5_reduce, 64, 5, 1)
        inception_4b_pool = self.max_pool('inception_4b_pool', inception_4a_output, 3, 1)
        inception_4b_pool_proj = self.conv2d('inception_4b_pool_proj', inception_4b_pool, 64, 1, 1)
        inception_4b_output = self.concat('inception_4b_output', [inception_4b_1x1, inception_4b_3x3, inception_4b_5x5,
                                                                  inception_4b_pool_proj])

        inception_4c_1x1 = self.conv2d('inception_4c_1x1', inception_4b_output, 128, 1, 1)
        inception_4c_3x3_reduce = self.conv2d('inception_4c_3x3_reduce', inception_4b_output, 128, 1, 1)
        inception_4c_3x3 = self.conv2d('inception_4c_3x3', inception_4c_3x3_reduce, 256, 3, 1)
        inception_4c_5x5_reduce = self.conv2d('inception_4c_5x5_reduce', inception_4b_output, 24, 1, 1)
        inception_4c_5x5 = self.conv2d('inception_4c_5x5', inception_4c_5x5_reduce, 64, 5, 1)
        inception_4c_pool = self.max_pool('inception_4c_pool', inception_4b_output, 3, 1)
        inception_4c_pool_proj = self.conv2d('inception_4c_pool_proj', inception_4c_pool, 64, 1, 1)
        inception_4c_output = self.concat('inception_4c_output', [inception_4c_1x1, inception_4c_3x3, inception_4c_5x5,
                                                                  inception_4c_pool_proj])

        inception_4d_1x1 = self.conv2d('inception_4d_1x1', inception_4c_output, 112, 1, 1)
        inception_4d_3x3_reduce = self.conv2d('inception_4d_3x3_reduce', inception_4c_output, 144, 1, 1)
        inception_4d_3x3 = self.conv2d('inception_4d_3x3', inception_4d_3x3_reduce, 288, 3, 1)
        inception_4d_5x5_reduce = self.conv2d('inception_4d_5x5_reduce', inception_4c_output, 32, 1, 1)
        inception_4d_5x5 = self.conv2d('inception_4d_5x5', inception_4d_5x5_reduce, 64, 5, 1)
        inception_4d_pool = self.max_pool('inception_4d_pool', inception_4c_output, 3, 1)
        inception_4d_pool_proj = self.conv2d('inception_4d_pool_proj', inception_4d_pool, 64, 1, 1)
        inception_4d_output = self.concat('inception_4d_output', [inception_4d_1x1, inception_4d_3x3, inception_4d_5x5,
                                                                  inception_4d_pool_proj])

        inception_4e_1x1 = self.conv2d('inception_4e_1x1', inception_4d_output, 256, 1, 1)
        inception_4e_3x3_reduce = self.conv2d('inception_4e_3x3_reduce', inception_4d_output, 160, 1, 1)
        inception_4e_3x3 = self.conv2d('inception_4e_3x3', inception_4e_3x3_reduce, 320, 3, 1)
        inception_4e_5x5_reduce = self.conv2d('inception_4e_5x5_reduce', inception_4d_output, 32, 1, 1)
        inception_4e_5x5 = self.conv2d('inception_4e_5x5', inception_4e_5x5_reduce, 128, 5, 1)
        inception_4e_pool = self.max_pool('inception_4e_pool', inception_4d_output, 3, 1)
        inception_4e_pool_proj = self.conv2d('inception_4e_pool_proj', inception_4e_pool, 128, 1, 1)
        inception_4e_output = self.concat('inception_4e_output', [inception_4e_1x1, inception_4e_3x3, inception_4e_5x5,
                                                                  inception_4e_pool_proj])

        pool4_3x3_s2 = self.max_pool('pool4_3x3_s2', inception_4e_output, 3, 2)
        inception_5a_1x1 = self.conv2d('inception_5a_1x1', pool4_3x3_s2, 256, 1, 1)
        inception_5a_3x3_reduce = self.conv2d('inception_5a_3x3_reduce', pool4_3x3_s2, 160, 1, 1)
        inception_5a_3x3 = self.conv2d('inception_5a_3x3', inception_5a_3x3_reduce, 320, 3, 1)
        inception_5a_5x5_reduce = self.conv2d('inception_5a_5x5_reduce', pool4_3x3_s2, 32, 1, 1)
        inception_5a_5x5 = self.conv2d('inception_5a_5x5', inception_5a_5x5_reduce, 128, 5, 1)
        inception_5a_pool = self.max_pool('inception_5a_pool', pool4_3x3_s2, 3, 1)
        inception_5a_pool_proj = self.conv2d('inception_5a_pool_proj', inception_5a_pool, 128, 1, 1)
        inception_5a_output = self.concat('inception_5a_output', [inception_5a_1x1, inception_5a_3x3, inception_5a_5x5,
                                                                  inception_5a_pool_proj])

        inception_5b_1x1 = self.conv2d('inception_5b_1x1', inception_5a_output, 384, 1, 1)
        inception_5b_3x3_reduce = self.conv2d('inception_5b_3x3_reduce', inception_5a_output, 192, 1, 1)
        inception_5b_3x3 = self.conv2d('inception_5b_3x3', inception_5b_3x3_reduce, 384, 3, 1)
        inception_5b_5x5_reduce = self.conv2d('inception_5b_5x5_reduce', inception_5a_output, 48, 1, 1)
        inception_5b_5x5 = self.conv2d('inception_5b_5x5', inception_5b_5x5_reduce, 128, 5, 1)
        inception_5b_pool = self.max_pool('inception_5b_pool', inception_5a_output, 3, 1)
        inception_5b_pool_proj = self.conv2d('inception_5b_pool_proj', inception_5b_pool, 128, 1, 1)
        inception_5b_output = self.concat('inception_5b_output', [inception_5b_1x1, inception_5b_3x3, inception_5b_5x5,
                                                                  inception_5b_pool_proj])

        pool5_7x7_s1 = self.avg_pool('pool5_7x7_s1', inception_5b_output, 7, 1)
        pool5_drop_7x7_s1 = self.dropout('pool5_drop_7x7_s1', pool5_7x7_s1, 0.6)

        self.logits = self.fc('loss3_classifier', pool5_drop_7x7_s1, out_nodes=self.config.n_classes)

        self.cal_loss(self.logits, self.y)
        self.cal_accuracy(self.logits, self.y)
        train_op = self.optimize()
        return train_op
