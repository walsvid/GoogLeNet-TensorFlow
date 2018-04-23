import tensorflow as tf
import numpy as np
from lib.utils.config import Config, TrainNetConfig


class Net:
    def __init__(self, cfg_):
        """
        :type cfg_: Config
        :param cfg_:
        """
        self.config = cfg_
        self.saver = None
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        self.scope = {}

    # save function thet save the checkpoint in the path defined in configfile
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiments path defined in config_file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def load_with_skip(self, data_path, session, skip_layer):
        data_dict = np.load(data_path, encoding='latin1').item()  # type: dict
        for key in data_dict.keys():
            if key not in skip_layer:
                # with tf.variable_scope(key, reuse=True, auxiliary_name_scope=False):
                with tf.variable_scope(self.scope[key], reuse=True) as scope:
                    with tf.name_scope(scope.original_name_scope):
                        for subkey, data in data_dict[key].items():
                            session.run(tf.get_variable(subkey).assign(data))

