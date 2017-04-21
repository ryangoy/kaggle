import sys
sys.path.insert(0, '../..')
import numpy as np
import time
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Activation, Lambda
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3, vgg16
from keras.applications.resnet50 import identity_block, conv_block
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import optimizers
from keras import backend as K
import paths

TRAIN_WEIGHTS = paths.IMAGENET_WEIGHTS_NO_TOP
TEST_WEIGHTS = paths.VGG16_RENTHOP_WEIGHTS

DROPOUT = .5

def VGG16_train(input_tensor=None, input_shape=(224, 224, 3),
          pooling=None, classes=1000):
   

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    model = Sequential()
    # Block 1
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # load weights
    model.load_weights(TRAIN_WEIGHTS)

    for layer in model.layers:
        layer.trainable = False

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(Dense(4096, activation='relu', name='pred'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    #model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0), metrics=["accuracy"])
    return model


def VGG16_test(weights_path, input_tensor=None, input_shape=(224, 224, 3),
          pooling=None, classes=1000):
   

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    model = Sequential()
    # Block 1
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Fully connected
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(Dense(4096, activation='relu', name='pred'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    # load weights
    model.load_weights(TEST_WEIGHTS)

    return model
