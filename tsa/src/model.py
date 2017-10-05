import numpy as np
import pandas as pd
import tensorflow as tf
from operator import mul


class TSANet:
  
  def __init__(self, input_shape, num_classes):

    # Defines self.x, self.y, self.model, self.cost, self.optimizer
    self.init_model(input_shape, num_classes)

  def conv2d(self, x, kshape, name='conv2d'):
    """
    Adds a conv2d layer to an unfinished graph x. 
 
    Args:
      x: an input tensor
      kshape: a tuple with shape [conv_width, conv_height, input_features, output_features]
      name: name of layer
    """
    W = tf.Variable(tf.truncated_normal(kshape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[kshape[-1]]))
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding ='SAME', name=name)
    relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    return relu
 
  def max_pool(self, x, kshape, name='conv2d'):
    """
    Args:
      x: an input tensor
      kshape: a tuple with shape [batch_step, width_step, height_step, feature_step]
      name: name of layer
    """
    return tf.nn.max_pool(x, ksize=kshape, strides=kshape, 
                          padding='SAME', name=name)

  def fc_layer(self, x, num_nodes, name='fc_layer', activation=tf.nn.relu):
    W = tf.Variable(tf.truncated_normal([x.shape[-1].value, num_nodes], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_nodes]))

    return tf.nn.bias_add(tf.matmul(x, W), b)

  def dropout(self, x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

  def init_model(self, input_shape, num_classes, num_lstm_hidden=512, learning_rate=0.01):
    """
    Args:
      input_shape: tuple [height, width, timesteps]
      num_classes: Number of label classes
    """
    timesteps = input_shape[-1]
    num_hidden_1 = 64
    num_hidden_2 = 128
    num_hidden_3 = 256
    num_hidden_4 = 512

    self.x = tf.placeholder(tf.float32, (None,) + input_shape)
    self.y = tf.placeholder(tf.float32, [None, num_classes])
    
    # Combine batch and timestep dimensions so we can run convolutions.
    x = tf.reshape(self.x, (-1,) + input_shape[-3:])
    x = self.conv2d(x, [3, 3, input_shape[-1], num_hidden_1], 'conv1_1')
    x = self.conv2d(x, [3, 3, num_hidden_1, num_hidden_1], 'conv1_2')
    x = self.max_pool(x, [1, 2, 2, 1], 'pool1')

    x = self.conv2d(x, [3, 3, num_hidden_1, num_hidden_2], 'conv2_1')
    x = self.conv2d(x, [3, 3, num_hidden_2, num_hidden_2], 'conv2_2')
    x = self.max_pool(x, [1, 2, 2, 1], 'pool2')

    x = self.conv2d(x, [3, 3, num_hidden_2, num_hidden_3], 'conv3_1')
    x = self.conv2d(x, [3, 3, num_hidden_3, num_hidden_3], 'conv3_2')
    x = self.conv2d(x, [3, 3, num_hidden_3, num_hidden_3], 'conv3_3')
    x = self.max_pool(x, [1, 2, 2, 1], 'pool3')

    x = self.conv2d(x, [3, 3, num_hidden_3, num_hidden_4], 'conv4_1')
    x = self.conv2d(x, [3, 3, num_hidden_4, num_hidden_4], 'conv4_2')
    x = self.conv2d(x, [3, 3, num_hidden_4, num_hidden_4], 'conv4_3')
    x = self.max_pool(x, [1, 2, 2, 1], 'pool4')

    x = self.conv2d(x, [3, 3, num_hidden_4, num_hidden_4], 'conv5_1')
    x = self.conv2d(x, [3, 3, num_hidden_4, num_hidden_4], 'conv5_2')
    x = self.conv2d(x, [3, 3, num_hidden_4, num_hidden_4], 'conv5_3')
    x = self.max_pool(x, [1, 2, 2, 1], 'pool5')
    
    conv_shape = x.shape[-3:].as_list()
    n_features = reduce(mul, conv_shape, 1)

    # Split batches and timestep dimension.
    x = tf.reshape(x, (-1, timesteps, n_features))

    forward_cell = tf.contrib.rnn.BasicLSTMCell(num_lstm_hidden)
    backward_cell = tf.contrib.rnn.BasicLSTMCell(num_lstm_hidden)

    x = tf.unstack(x, input_shape[0], 1)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(forward_cell, backward_cell,
                                                              x, dtype=tf.float32)

    # Linear activation.
    self.model = self.fc_layer(outputs[-1], num_classes)

    # Probability error for each class, which is assumed to be independent.
    self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.model))
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

  def train_val(self, X_trn, y_trn, X_val=None, y_val=None, epochs=10, batch_size=1,
                display_step=100):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      for epoch in range(epochs):
        for step in range(int(X_trn.shape[0]/batch_size)):
          batch_x, batch_y = self.get_next_batch(X_trn, y_trn, batch_size)
          sess.run(self.optimizer, feed_dict={x: batch_x, y: batch_y})

          if step % display_step == 0:
            loss, acc = sess.run([self.cost, self.accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iter {}, Minibatch Loss={:.6f}, Training Accuracy={:.5f}.".format(step, loss, acc))

          
        if X_val is not None and y_val is not None:
          loss = sess.run(self.cost, feed_dict={x: X_val, y: y_val})

        print("Epoch {}, Validation Loss={:6f}, Validation Accuracy={:.5f}.".format(loss, acc))



if __name__ == '__main__':
  TSANet((64, 512, 512, 1), 16)
