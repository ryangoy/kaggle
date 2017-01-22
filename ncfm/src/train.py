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

    X = None
    for i in range(num_images):
        img = utils.load_image("./preprocessed_train/img_{0}label_{1}.jpg".format(i, y_index[i]))
        img = img.reshape((1, 224, 224, 3))
        if X == None:
            X = img
        else:
            X = np.vstack((X, img))
    train_net(X, y)
    return


if __name__ == '__main__':
    run()