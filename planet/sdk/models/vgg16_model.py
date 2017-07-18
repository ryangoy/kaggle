import numpy as np
import pandas as pd
import math
from model import Model
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import *

SEED = 0
VGG_WEIGHTS = '/home/ryan/cs/datasets/planet/weights/vgg16_no_top.h5'

class VGG16(Model):
    """
    VGG16 model. Uses Keras's 
    """
    def __init__(self, name='VGG16', labels=None, num_epochs=1,
                 batch_size=32, classes=2, verbose=1, 
                 num_train_images=40479, num_test_images=61191,
                 augmentation_params={}):
        self.model = init_model(classes=classes, input_shape=(224, 224, 3))
        self.name = name
        self.labels = labels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        if 'featurewise_center' in augmentation_params.keys() or \
           'featurewise_std_normalization' in augmentation_params.keys() or \
           'zca_whitening' in augmentation_params.keys():
            print """Certain features for Keras data augmentation require all 
                  data to be loaded up front. Please disable and manually 
                  implement the following features:
                  featurewise_center or featurewise_std_normalization or zca_whitening"""
        self.train_gen = image.ImageDataGenerator(**augmentation_params, rescale=1. / 255)
        self.test_gen = image.ImageDataGenerator(rescale=1. / 255)
        self.num_classes = classes
        self.verbose = verbose
        self.num_test_images = num_test_images
        self.num_train_images = num_train_images
        self.steps_per_epoch = num_train_images / batch_size/100+ 1 


    def train(self, X_trn, y_trn, X_val=None, y_val=None):
        trn_generator = self.train_gen.flow_from_directory(X_trn, batch_size=self.batch_size,
                                                    target_size = (224, 224))
        #TODO make val_generator without data augmentation
        val_generator = None
        if X_val is not None:
            val_generator = self.test_gen.flow_from_directory(X_val, batch_size=self.batch_size,
                                                        target_size = (224, 224))
        history = self.model.fit_generator(trn_generator, self.steps_per_epoch, 
                                            epochs=self.num_epochs, verbose=self.verbose,
                                            validation_data = val_generator, 
                                            validation_steps = self.steps_per_epoch / 10)
        return history

    def test(self, X_test, y_test=None):
        test_generator = self.test_gen.flow_from_directory(X_test, batch_size=self.batch_size, target_size = (224, 224))
        num_steps = math.ceil(float(self.num_test_images)/self.batch_size/1000+1)
        predictions = self.model.predict_generator(test_generator, steps=num_steps)
        return predictions[:,0]

def init_model(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):

    # Block 1
    model = Sequential()
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Does not load the FC layers
    model.load_weights(VGG_WEIGHTS)

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(classes, activation='sigmoid', name='predictions'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
        
    return model
   
