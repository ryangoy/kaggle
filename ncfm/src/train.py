from vgg19 import VGG19
import tensorflow as tf
import utils
import numpy as np


epochs = 10
batch_size = 100

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

def train_net(batch, labels):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, 6])
    train_mode = tf.placeholder(tf.bool)

    #vgg = VGG19()
    vgg = VGG19('/home/ryan/cs/datasets/ncfm/vgg19.npy')
    #vgg = VGG19('./src/vgg19.npy')
    print "Building..."
    vgg.build(images, train_mode)
    print "Initializing variables"
    sess.run(tf.initialize_all_variables())

    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    print "Training..."

    for epoch in range(epochs):
        for i in range(num_images // batch_size):
            sess.run(train, feed_dict={images: batch[batch_size*i:batch_size*(i+1)], 
                     true_out: labels[batch_size*i:batch_size*(i+1)], train_mode: True})
        print 'Finished epoch ' + str(epoch)



    # for epoch in range(num_epochs):
    #     for i in range(num_images // batch_size):
    #         sess.run(train, feed_dict={images: batch[batch_size*i:batch_size*(i+1)], 
    #                  true_out: labels[batch_size*i:batch_size*(i+1)], train_mode: True})
    #     cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    #     _, loss_val = sess.run([train, cross_entropy],
    #                        feed_dict={x: batch, y_: labels})
    #     print epoch, loss_val

    # prob = sess.run(vgg.prob, feed_dict={images: batch[:100], train_mode: False})
    # print np.argmax(prob, axis=1)

    print "Saving new model..."
    vgg.save_npy(sess, './test-save.npy')
    sess.close()
    print "Done!"

def test_net(batch):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, 6])
    train_mode = tf.placeholder(tf.bool)

    vgg = VGG19('./test-save.npy')
    print "Building..."
    vgg.build(images, train_mode)
    print "Initializing variables"
    sess.run(tf.initialize_all_variables())

    prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
    sess.close()
    print np.argmax(prob, axis=1)

def load_data():
    y = []
    y_index = []
    for i in range(len(label_counts)):
        y += [labels[i] for _ in range(label_counts[i])]
        y_index += [i for _ in range(label_counts[i])]
    y = np.array(y)

    X = np.empty((num_images, 224, 224, 3))
    for i in range(num_images):
        img = utils.load_image("/home/ryan/cs/kaggle/ncfm/preprocessed_train/img_{0}label_{1}.jpg".format(i, y_index[i]))
        #img = utils.load_image("/home/mzhao/Desktop/kaggle/ncfm/preprocessed_train/img_{0}label_{1}.jpg".format(i, y_index[i]))
        img = img.reshape((1, 224, 224, 3))
        X[i] = img
        if i % 1000 == 0 and i > 0:
            print "Finished pre-processing " + str(i) + " images"
    print X.shape
    print y.shape
    return X, y

def run():
    batch, labels = load_data()
    #train_net(batch, labels)
    test_net(batch[:100])
    print np.argmax(labels[:100], axis=1)



if __name__ == '__main__':
    run()