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
import matplotlib.pyplot as plt

def vgg16(size=(270, 480), lr=0.01, dropout=0.4, nb_classes=8):
    # px_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
    px_mean = np.array([98.428,107.430,96.967]).reshape((3,1,1))

    def vgg_preprocess(x):
        x = x - px_mean
        return x[:, ::-1]
    #     return tf.reverse(x, [False, True, False, False]) # for tensorflow only
        return x

    weights_file='weights/vgg16.h5'

    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3,)+size, output_shape=(3,)+size))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights(weights_file)
    model.pop(); model.pop(); model.pop()

    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(4096, activation='relu', name='pred'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, activation='softmax', name='predictions'))

    # model.compile(loss='categorical_crossentropy', optimizer="nadam", metrics=["accuracy"])
    # model.compile(loss='categorical_crossentropy', optimizer="adamax", metrics=["accuracy"])
    # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
    # model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0), metrics=["accuracy"])
    # model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=lr, momentum=0.0, decay=0.1, nesterov=True), metrics=["accuracy"])
    return model

def vgg16_bb(size=(270, 480), lr=0.001, dropout=0.4, nb_classes=8, output='regression'):
    # px_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
    px_mean = np.array([98.428,107.430,96.967]).reshape((3,1,1))

    def vgg_preprocess(x):
        x = x - px_mean
        return x[:, ::-1]
    #     return tf.reverse(x, [False, True, False, False]) # for tensorflow only
        return x

    weights_file='weights/vgg16.h5'

    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3,)+size))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights(weights_file)
    model.pop(); model.pop(); model.pop()

    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(4096, activation='relu', name='pred'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    if output == 'regression':
        model.add(Dense(4, activation='linear', name='predictions'))
        # model.compile(loss='mse', optimizer="nadam", metrics=["accuracy"])
        # model.compile(loss='mse', optimizer="adamax", metrics=["accuracy"])
        # model.compile(loss='mse', optimizer="adam", metrics=["accuracy"])
        # model.compile(loss='mse', optimizer='adadelta', metrics=["accuracy"])
        model.compile(loss='mse', optimizer='adagrad', metrics=["accuracy"])
    elif output == 'classification':
        model.add(Dense(2*size[0]+2*size[1], activation='sigmoid', name='predictions'))
        # model.compile(loss='binary_crossentropy', optimizer="nadam", metrics=["accuracy"])
        # model.compile(loss='binary_crossentropy', optimizer="adamax", metrics=["accuracy"])
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
        # model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=["accuracy"])
        # model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=["accuracy"])

    return model

def mean_error(y_true, y_pred):
    return K.mean(y_true-y_pred)


def aligner_vgg16(weights_file='weights/vgg16.h5', size=(270, 480), lr=0.001, dropout=0.4):

    model = Sequential()
    #model.add(Lambda(vgg_preprocess, input_shape=size+(3,)))
    model.add(ZeroPadding2D((1,1), input_shape=(3,)+size))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights(weights_file)
    model.pop(); model.pop(); model.pop()

    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(4096, activation='relu', name='pred'))
    # model.add(BatchNormalization())
    # model.add(Dropout(dropout))

    model.add(Dense(1, name='predictions'))
    model.compile(loss='mse', optimizer='adam')
    return model



def train_all(model, trn_all_gen, nb_trn_all_samples=3777,
              nb_epoch=10, weightfile='default.h5'):
    history = model.fit_generator(trn_all_gen, samples_per_epoch=nb_trn_all_samples, nb_epoch=nb_epoch, verbose=2)
    model.save_weights('weights/train_all/{}'.format(weightfile))


def train_val(model, trn_gen, val_gen, nb_trn_samples=3207, nb_val_samples=570,
              nb_epoch=12, weightfile='default.h5'):
    history = model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                validation_data=val_gen, nb_val_samples=nb_val_samples)
    # model.save_weights('weights/train_val/{}'.format(weightfile))
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def predict(model, test_gen_fn=None, predfile='default.npy', nb_test_samples=1000, nb_classes=8, nb_runs=5, nb_aug=5):
    start_time = time.time()        

    predictions_full = np.zeros((nb_test_samples, nb_classes))
    
    for run in range(nb_runs):
        print("\nStarting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
        predictions_aug = np.zeros((nb_test_samples, nb_classes))
        
        for aug in range(nb_aug):
            print("\n--Predicting on Augmentation {0} of {1}...\n".format(aug+1, nb_aug))
            # make this more efficient instead of reading 1k images every time
            if test_gen_fn == None:
                test_gen = get_test_gens
            else:
                test_gen = test_gen_fn()
            predictions_aug += model.predict_generator(test_gen, val_samples=nb_test_samples)

        predictions_aug /= nb_aug
        predictions_full += predictions_aug

        print '{} runs in {} sec'.format(run+1, time.time() - start_time)
    
    predictions_full /= nb_runs
    np.save("pred/{}".format(predfile), predictions_full)

def get_train_all_gens(X=None, y=None, size=(270, 480), batch_size=16):
    trn_all_path = 'train_all/'
    
    trn_all_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      vertical_flip=True)
    if X is not None and y is not None:
        trn_all_gen = trn_all_datagen.flow(X, y, batch_size=batch_size,
                                                        shuffle=True)
    else:
        trn_all_gen = trn_all_datagen.flow_from_directory(trn_all_path, target_size=size, batch_size=batch_size,
                                                        class_mode='categorical', shuffle=True)
    return trn_all_gen

def get_train_val_gens(X_trn=None, X_val=None, y_trn=None, y_val=None, size=(270, 480), batch_size=16):
    trn_path = 'train/'
    val_path = 'valid/'

    trn_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      vertical_flip=True)
    val_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      vertical_flip=True)
    if X_trn is not None and y_trn is not None:
        trn_gen = trn_datagen.flow(X_trn, y_trn, batch_size=batch_size,
                                                        shuffle=True)
    else:
        trn_gen = trn_datagen.flow_from_directory(trn_path, target_size=size, batch_size=batch_size,
                                                        class_mode='categorical', shuffle=True)
    if X_val is not None and y_val is not None:
        val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size,
                                                           shuffle=False)
    else:
        val_gen = val_datagen.flow_from_directory(val_path, target_size=size, batch_size=batch_size,
                                                           class_mode='categorical', shuffle=False)
    return trn_gen, val_gen

def get_test_gens(X_test=None, size=(270, 480), batch_size=16, test_path='test/'):
    # test_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      vertical_flip=True)
    if X_test is not None:
        test_gen = test_datagen.flow(X_test, batch_size=batch_size,
                                                        shuffle=False)
    else:
        test_gen = test_datagen.flow_from_directory(test_path, target_size=size, batch_size=batch_size,
                                                    class_mode='categorical', shuffle=False)
    return test_gen









