import argparse
import keras
import numpy as np
from pipeline.train import train_kfold
from pipeline.load_data import load_trn_data, load_test_data, load_trn_metadata, load_test_metadata
# from model.resnet import ResNetModel
# from model.naive_res import NaiveResNetModel, create_network, cosine_annealing, iou_bce_loss, mean_iou
import tensorflow as tf

TRN_IMAGES_PATH = '/home/ryan/cs/datasets/rsna/stage_1_train_images'
TRN_LABELS_PATH = '/home/ryan/cs/datasets/rsna/stage_1_train_labels.csv'


def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=4):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2**depth)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# define iou or jaccard loss function# defin 
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))



# cosine learning rate annealing
def cosine_annealing(x):
    lr = 0.001
    epochs = 25
    return lr*(np.cos(np.pi*x/epochs)+1.)/2

def main(args):
	trn_gen, val_gen = load_trn_data(TRN_IMAGES_PATH, TRN_LABELS_PATH)

	# model = NaiveResNetModel((256, 256, 1), 32)

	# model.train(trn_gen, val_gen, epochs=10)

	model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
	model.compile(optimizer='adam',
	              loss=iou_bce_loss,
	              metrics=['accuracy', mean_iou])

	learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)

	history = model.fit_generator(trn_gen, validation_data=val_gen, callbacks=[learning_rate], epochs=25, workers=4, use_multiprocessing=True)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-e', '--epochs', default='10', type=int)

	args = parser.parse_args()

	main(args)