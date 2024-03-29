from vgg19 import VGG19
import tensorflow as tf
import utils
import numpy as np
import pandas as pd
import argparse
import os.path
import sys
import time
import skimage.io
from load_fish import load_processed_fish
import paths

# Basic model parameters as external flags.
FLAGS = None

# placeholders initialize the size of the input and output
def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, 
        shape=[batch_size, FLAGS.width, FLAGS.height, FLAGS.channels])
    #labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size, FLAGS.num_categories])
    labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size])
    train_mode = tf.placeholder(tf.bool)
    return images_placeholder, labels_placeholder, train_mode

def loss_op(logits, labels):

    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #    labels=labels, logits=logits, name='xentropy')
    #return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    # labels = tf.cast(labels, tf.float32)  
    # # l2_function = tf.nn.l2_loss(logits - labels)
    # l2_function = tf.nn.l2_loss(tf.log(logits) * labels)
    # loss = tf.reduce_sum(l2_function, name='l2_loss')

    loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    return loss

def training_op(loss, learning_rate):

    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation_op(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

# sess.run() uses feed_dicts to train
def fill_feed_dict(step, train_mode, train_mode_pl,
    images, images_pl, labels=None, labels_pl=None):

    images_feed = images[FLAGS.batch_size*step:FLAGS.batch_size*(step+1)]
    feed_dict = {images_pl: images_feed, train_mode_pl: train_mode}

    if train_mode: # if we're training, we'll also need labels

        labels_feed = labels[FLAGS.batch_size*step:FLAGS.batch_size*(step+1)]
        feed_dict[labels_pl] = labels_feed

    return feed_dict

def do_eval(sess, eval_correct,
            images_placeholder, images,
            labels_placeholder, labels,
            train_mode_placeholder):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    num_examples = images.shape[0]
    for i in xrange(num_examples // FLAGS.batch_size):
        feed_dict = {images_placeholder: images[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)],
                     labels_placeholder: labels[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)],
                     train_mode_placeholder: False,}
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def run_training():

    tf.set_random_seed(1)
    
    # initialize placeholders, tm stands for train mode
    images_pl, labels_pl, tm_pl = placeholder_inputs(FLAGS.batch_size)

    # load data
    print "Loading data..."
    training_images, training_labels, val_images, val_labels = load_processed_fish(FLAGS.images_path, 500)

    # create VGG19 model
    vgg = VGG19(FLAGS.vgg_path)
    vgg.build(images_pl, tm_pl)

    # define loss with output of vgg net and labels
    loss = loss_op(vgg.prob, labels_pl) 

    # define training operation
    train_op = training_op(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation_op(vgg.prob, labels_pl)

    # for display purposes later
    summary = tf.summary.merge_all()

    # create a saver for training checkpoints
    saver = tf.train.Saver()
    
    # begin session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # load pre-trained weights
    print "Initializing..."
    sess.run(tf.global_variables_initializer())

    print "Training..."
    training_start_time = time.time()
    step = 0
    num_batches = training_images.shape[0] // FLAGS.batch_size
    for epoch in range(FLAGS.num_epochs):
        epoch_start_time = time.time()
        for i in range(num_batches):
            step += 1
            feed_dict = {images_pl: training_images[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)],
                           labels_pl: training_labels[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)],
                           tm_pl: True}
            _, loss_value, guesses, labs = sess.run([train_op, loss, vgg.prob, labels_pl], 
                feed_dict=feed_dict)
            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if step % 10 == 0:
                # print "Guesses: " + str(np.argmax(guesses, axis=1))
                # print "Results: "+ str(labs)
                # idk = np.argmax(guesses,axis=1)
                # num_sim = sum([idk[i] == labs[i] for i in range(len(labs))])
                # print "accuracy: %.2f" % (float(num_sim) / len(labs))
                print('Step %d: loss = %.5f (%.2f min)' % (step, loss_value, (time.time() - training_start_time)/60))
        print '[Epoch %d] loss = %.5f (%.2f min)' % (epoch, loss_value, (time.time() - training_start_time)/60)
        print 'Training Data Eval:'
        do_eval(sess, eval_correct,
                images_pl, training_images,
                labels_pl, training_labels,
                tm_pl)
        print 'Validation Data Eval:'
        do_eval(sess, eval_correct,
                images_pl, val_images,
                labels_pl, val_labels,
                tm_pl)
    print "Saving new model..."
    vgg.save_npy(sess, './test-save.npy')
    sess.close()
    print "Done!"


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    np.random.seed(seed=0)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0003,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--vgg_path',
        type=str,
        default=paths.vgg_path,
        help='Directory to the pre-trained vgg model.'
    )
    parser.add_argument(
        '--images_path',
        type=str,
        default=paths.images_path,
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=paths.log_dir,
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
        default=100,
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