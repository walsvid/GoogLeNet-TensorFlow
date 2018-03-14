import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from lib.data_loader.data_loader import Flowers102DataLoader
from lib.utils.config import ConfigReader, TrainNetConfig, DataConfig
from lib.googlenet.inception_v1 import InceptionV1


def plot_image_test(image_batch, label_batch, train_config):
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
                img, label = sess.run([image_batch, label_batch])
                # just test one batch
                for j in np.arange(train_config.batch_size):
                    print('label: %d' % label[j])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                    pass
                i += 1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)


def train():
    config_reader = ConfigReader('experiments/configs/inception_v1.yml')
    train_config = TrainNetConfig(config_reader.get_train_config())
    data_config = DataConfig(config_reader.get_train_config())

    train_log_dir = './logs/train/'
    val_log_dir = './logs/val/'

    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir):
        os.makedirs(val_log_dir)

    net = InceptionV1(train_config)

    with tf.name_scope('input'):
        train_loader = Flowers102DataLoader(data_config, is_train=True, is_shuffle=True)
        train_image_batch, train_label_batch = train_loader.generate_batch()
        val_loader = Flowers102DataLoader(data_config, is_train=False, is_shuffle=False)
        val_image_batch, val_label_batch = val_loader.generate_batch()

    train_op = net.build_model()
    summaries = net.get_summary()

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge(summaries)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    net.load_with_skip(train_config.pre_train_weight, sess, ['loss3_classifier'])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(train_config.max_step):
            if coord.should_stop():
                break

            train_image, train_label = sess.run([train_image_batch, train_label_batch])
            _, train_loss, train_acc = sess.run([train_op, net.loss, net.accuracy],
                                                feed_dict={net.x: train_image, net.y: train_label})

            if step % 50 == 0 or step + 1 == train_config.max_step:
                print('===TRAIN===: Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: train_image, net.y: train_label})
                train_summary_writer.add_summary(summary_str, step)
            if step % 200 == 0 or step + 1 == train_config.max_step:
                val_image, val_label = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([net.loss, net.accuracy], feed_dict={net.x: val_image, net.y: val_label})
                print('====VAL====: Step %d, val loss = %.4f, val accuracy = %.4f%%' % (step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: train_image, net.y: train_label})
                val_summary_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or step + 1 == train_config.max_step:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('===INFO====: Training completed, reaching the maximum number of steps')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()
