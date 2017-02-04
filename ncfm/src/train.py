from vgg19 import VGG19
import tensorflow as tf
import utils
import numpy as np
import argparse
import os.path
import sys
import time

# Basic model parameters as external flags.
FLAGS = None

# placeholders initialize the size of the input and output
def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, 
        shape=[batch_size, FLAGS.width, FLAGS.height, FLAGS.channels])
    labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size, FLAGS.num_categories])
    train_mode = tf.placeholder(tf.bool)
    return images_placeholder, labels_placeholder, train_mode

# sess.run() uses feed_dicts to train
def fill_feed_dict(step, train_mode, train_mode_pl,
    images, images_pl, labels=None, labels_pl=None):

    images_feed = images[FLAGS.batch_size*step:FLAGS.batch_size*(step+1)]
    feed_dict = {images_pl: images_feed, train_mode_pl: train_mode}

    if train_mode: # if we're training, we'll also need labels
        labels_feed = labels[FLAGS.batch_size*step:FLAGS.batch_size*(step+1)]
        feed_dict[labels_pl] = labels_feed

    return feed_dict

def loss_op(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.cast(labels, tf.float32)
  #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  #    labels=labels, logits=logits, name='xentropy')
  #return tf.reduce_mean(cross_entropy, name='xentropy_mean')
  l2_loss = (logits - labels) ** 2
  return tf.reduce_sum(l2_loss, name='l2_loss')

def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def run_training():
    
    # initialize placeholders, tm stands for train mode
    images_pl, labels_pl, tm_pl = placeholder_inputs(FLAGS.batch_size)

    # load data
    images, labels = load_data()
    num_images = images.shape[0]

    # create VGG19 model
    vgg = VGG19('/home/ryan/cs/datasets/ncfm/vgg19.npy')
    vgg.build(images_pl, tm_pl)
    logits = vgg.prob # set logits to the softmax output of the net

    # call the above methods to return the training optimizer
    loss = loss_op(logits, labels_pl)
    train_op = training(loss, FLAGS.learning_rate)

    # # Add the Op to compare the logits to the labels during evaluation.
    # eval_correct = mnist.evaluation(logits, labels_placeholder)

    # for display purposes later
    summary = tf.summary.merge_all()

    # create a saver for training checkpoints
    saver = tf.train.Saver()
    
    # begin session
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # load pre-trained weights
    print "Initializing..."
    sess.run(tf.global_variables_initializer())

    print "Training..."
    training_start_time = time.time()
    step = 0
    for epoch in range(FLAGS.num_epochs):
        epoch_start_time = time.time()
        for i in range(num_images // FLAGS.batch_size):
            step += 1
            feed_dict = fill_feed_dict(i, True, tm_pl, images,
                images_pl, labels, labels_pl)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.2f min)' % (step, loss_value, (time.time() - training_start_time)/60))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            elif step % 10 == 0:
                print('Step %d: loss = %.2f (%.2f min)' % (step, loss_value, (time.time() - training_start_time)/60))


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

    num_images = 3000
    label_bounds = [0, 199, 1910, 2641, 2758, 2933, 3000] #not using other fish
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

    X = np.empty((num_images, 224, 224, 3))
    for i in range(num_images):
        img = utils.load_image(os.path.join(FLAGS.images_path, "img_{0}label_{1}.jpg".format(i, y_index[i])))
        
        img = img.reshape((1, 224, 224, 3))
        X[i] = img
        if i % 1000 == 0 and i > 0:
            print "Finished pre-processing " + str(i) + " images"
    print "Loaded training images with shape {0}.".format(X.shape)
    print "Loaded training labels with shape {0}.".format(y.shape)
    return X, y


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--images_path',
        type=str,
        default='/home/ryan/cs/kaggle/ncfm/preprocessed_train',
        #'/home/mzhao/Desktop/kaggle/ncfm/preprocessed_train'
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/home/ryan/cs/kaggle/ncfm/logs',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='Number of channels in the input images (e.g. for RGB, channels = 3).'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=224,
        help='Width of input images in pixels.'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=224,
        help='Width of input images in pixels.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of epochs to run training for.'
    )
    parser.add_argument(
        '--num_categories',
        type=int,
        default=6,
        help='Number of categories.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)