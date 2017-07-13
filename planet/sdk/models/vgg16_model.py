import numpy as np
import pandas as pd
import math
from model import Model
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import *

SEED = 0
VGG_WEIGHTS = '/home/ryan/cs/datasets/planet/weights/vgg16_no_top.h5'

class VGG16(Model):
    """
    VGG16 model. Uses Keras's 
    """
    def __init__(self, train_dir, test_dir, name='VGG16', labels=None, num_epochs=10,
                 batch_size=32, classes=None, verbose=1, 
                 num_test_images=40479, num_test_images= 61191,
                 augmentation_params={}):
        self.model = init_model(classes=classes)
        self.name = name
        self.labels = labels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        if 'featurewise_center' in dict.keys() or 
           'featurewise_std_normalization' in dict.keys() or 
           'zca_whitening' in dict.keys():
            print """Certain features for Keras data augmentation require all 
                  data to be loaded up front. Please disable and manually 
                  implement the following features:
                  featurewise_center or featurewise_std_normalization or zca_whitening"""
        self.gen = image.ImageDataGenerator(**augmentation_params)
        self.num_classes = len(self.gen.class_indices.keys())
        self.verbose = verbose
        self.num_test_images = num_test_images
        self.num_train_images = num_train_images


    def train(self, X_trn, y_trn, X_val=None, y_val=None):
        trn_generator = self.gen.flow_from_directory(X_trn, batch_size=self.batch_size)
        #TODO make val_generator without data augmentation
        val_generator = None
        if X_val is not None:
            val_generator = self.gen.flow_from_directory(X_val, batch_size=self.batch_size)
        history = self.model.fit_generator(trn_generator, STEPS_PER_EPOCH, 
                                            epochs=self.num_epochs, self.verbose,
                                            validation_data = val_generator, 
                                            validation_steps = STEPS_PER_EPOCH / 10)
        return history

    def test(self, X_test, y_test=None):
        test_generator = self.gen.flow_from_directory(X_test, batch_size=self.batch_size)
        num_steps = math.ceil(float(self.num_test_images)/self.batch_size+1)
        predictions = self.model.predict_generator(test_generator, steps=num_steps)
        return predictions

    def init_model(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):

        img_input = Input(shape=input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        model = Model(inputs, x, name='vgg16')

        # Does not load the FC layers
        model.load_weights(VGG_WEIGHTS)

        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dense(classes, activation='softmax', name='predictions'))
            
        return model

    