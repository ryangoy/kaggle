from vgg19 import VGG19
import tensorflow as tf
import utils


def train_net(batch, labels):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, 6])
    train_mode = tf.placeholder(tf.bool)

    vgg = VGG19('/home/ryan/cs/datasets/ncfm/vgg19.npy')
    vgg.build(images, train_mode)
    
    sess.run(tf.initialize_all_variables())

    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch, true_out: labels, train_mode: True})

    vgg.save_npy(sess, './test-save.npy')
    return

def run():
    
    train_net()
    return


if __name__ == '__main__':
    run()