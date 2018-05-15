import argparse
import tensorflow as tf
from scipy.misc import imread, imresize
from lib.utils.config import ConfigReader, TestNetConfig
from lib.googlenet.inception_v1 import InceptionV1
from lib.googlenet.inception_v2 import InceptionV2


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='VGG test demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [googlenet incption v1]',
                        default='InceptionV1')
    parser.add_argument('--im', dest='im_path', help='Path to the image',
                        default='data/demo/demo.jpg', type=str)
    parser.add_argument('--model', dest='model', help='Model path', default='./')
    parser.add_argument('--meta', dest='meta', help='Dataset meta info, class names',
                        default='./data/datasets/meta.txt', type=str)
    args = parser.parse_args()

    return args


def test_net():
    args = parse_args()
    config_reader = ConfigReader('experiments/configs/inception_v1.yml')
    test_config = TestNetConfig(config_reader.get_test_config())

    mode = 'RGB' if test_config.image_depth == 3 else 'L'
    img = imread(args.im_path, mode=mode)
    img = imresize(img, [test_config.image_height, test_config.image_width])  # height, width
    k = 1  # select top k
    with open(args.meta) as mf:
        class_names = mf.read().splitlines()
        class_names = list(filter(None, class_names))

    net = InceptionV1(test_config)
    logits = net.build_model()
    values, indices = net.cal_result(logits)

    ckpt_path = test_config.model_path
    # start a session
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    print('Model checkpoint path: {}'.format(ckpt_path))
    try:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except FileNotFoundError:
        raise 'Check your pretrained {:s}'.format(ckpt_path)

    [prob, ind, out] = sess.run([values, indices, logits], feed_dict={net.x: [img]})
    prob = prob[0]
    ind = ind[0]
    print('Classification Result:')
    for i in range(k):
        print('Category Name: %s \nProbability: %.2f%%' % (class_names[ind[i]], prob[i] * 100))
    sess.close()


if __name__ == '__main__':
    test_net()
