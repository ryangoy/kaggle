from vgg19 import VGG19
import tensorflow as tf
import utils
import numpy as np


def train_net(batch, labels):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, 6])
    train_mode = tf.placeholder(tf.bool)

    vgg = VGG19('/home/ryan/cs/datasets/ncfm/vgg19.npy')
    print "Building..."
    vgg.build(images, train_mode)
    print "Initializing variables"
    sess.run(tf.initialize_all_variables())

    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    print "Training..."
    sess.run(train, feed_dict={images: batch, true_out: labels, train_mode: True})
    print "Saving new model..."
    vgg.save_npy(sess, './test-save.npy')
    print "Done!"

def load_data():
    num_images = 3000
    # num_images = 3299
    label_bounds = [0, 199, 1910, 2641, 2758, 2933, 3000] #not using other fish
    # label_bounds = [0, 199, 1910, 2641, 2758, 2933, 3000, 3299]
    label_counts = [label_bounds[i+1]-label_bounds[i] for i in range(len(label_bounds)-1)]
    labels = [[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]]
    y = []
    y_index = []
    for i in range(len(label_counts)):
        y += [labels[i] for _ in range(label_counts[i])]
        y_index += [i for _ in range(label_counts[i])]
    y = np.array(y)

    X = []
    for i in range(num_images):
        img = utils.load_image("/home/ryan/cs/kaggle/ncfm/preprocessed_train/img_{0}label_{1}.jpg".format(i, y_index[i]))
        img = img.reshape((224, 224, 3))
        X.append(img)
        if i % 100 == 0 and i > 0:
            print "Finished pre-processing " + str(i) + " images"
    print np.array(X).shape
    print y.shape
    return np.array(X), y

def run():
    train_net(*load_data())


if __name__ == '__main__':
    run()