import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
import vis

seed = 0
np.random.seed(seed)

import tensorflow as tf
import theano

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Activation, Lambda
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.applications.resnet50 import identity_block, conv_block
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import optimizers
from keras import backend as K

valid_percent = .15

vgg_size = (270, 480)
inception_size = (299, 299)

# fish_types = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
# fish_counts = [1745,202,117,68,442,286,177,740]
# fish_cumulative_counts = [0,1745,1947,2064,2132,2574,2860,3037,3777]
# nb_trn_all_samples = 3777
# nb_trn_samples = 3207
# nb_val_samples = 570


fish_types = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
fish_counts = [1745,202,117,68,286,177,740]
fish_cumulative_counts = [0] + [sum(fish_counts[:i+1]) for i in range(len(fish_counts))]
nb_trn_all_samples = fish_cumulative_counts[-1]
nb_trn_samples = int(sum([((1 - valid_percent)*100*c)//100 for c in fish_counts]))
nb_val_samples = nb_trn_all_samples - nb_trn_samples
# print fish_cumulative_counts
# print nb_trn_all_samples, nb_trn_samples, nb_val_samples

nb_test_samples = 1000
nb_classes = len(fish_types)

def load_data(saved=False):
    if not saved:
        X = []
        y = []
        index = 0
        for i in range(len(fish_types)):
            fish = fish_types[i]
            for file in os.listdir("train_all/{}".format(fish)):
                path = "train_all/{}/{}".format(fish, file)
                # img = np.array(keras.preprocessing.image.load_img(path, target_size=vgg_size))
                # img = skimage.io.imread(path)
                img = cv2.imread(path)
                # img = skimage.transform.resize(img, vgg_size).transpose((2, 0, 1))
                # print img.shape
                img = cv2.resize(img, (vgg_size[1], vgg_size[0]), cv2.INTER_LINEAR)
                # print img.shape
                img = img.transpose((2, 0, 1))
                # print img.shape
                X += [img]
                label = [0 for _ in range(len(fish_types))]
                label[i] = 1
                y += [label]
        X = np.array(X)
        y = np.array(y)
        print X.shape, y.shape
        np.save("X_fish_only.npy", X)
        np.save("y_fish_only.npy", y)
    else:
        X = np.load('X_fish_only.npy')
        y = np.load('y_fish_only.npy')
    # X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=.15)

    X_trn = []
    X_val = []
    y_trn = []
    y_val = []
    for i in range(len(fish_counts)):
        Xt, Xv, yt, yv = train_test_split(X[fish_cumulative_counts[i]:fish_cumulative_counts[i+1]], 
                                          y[fish_cumulative_counts[i]:fish_cumulative_counts[i+1]], 
                                          test_size=valid_percent)
        X_trn += list(Xt)
        X_val += list(Xv)
        y_trn += list(yt)
        y_val += list(yv)
    X_trn = np.array(X_trn)
    X_val = np.array(X_val)
    y_trn = np.array(y_trn)
    y_val = np.array(y_val)
    # print X_trn.shape, y_trn.shape, X_val.shape, y_val.shape
    # print np.sum(y_trn, axis=0), np.sum(y_val, axis=0)
    # exit(1)
    return X, X_trn, X_val, y, y_trn, y_val

def vgg16():
    px_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))

    def vgg_preprocess(x):
        x = x - px_mean
        return x[:, ::-1]
    #     return tf.reverse(x, [False, True, False, False]) # for tensorflow only

    size = vgg_size
    lr = 0.001
    dropout = 0.4
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
    model.add(Dense(nb_classes, activation='softmax', name='predictions'))

    model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=["accuracy"])
    return model

def get_vgg_gens(X=None, X_trn=None, X_val=None, y=None, y_trn=None, y_val=None, train=True, test=True):
    trn_all_path = 'train_all/'
    trn_path = 'train/'
    val_path = 'valid/'
    test_path = 'test/'

    batch_size = 16
    size = vgg_size

    trn_all_gen = None
    trn_gen = None
    val_gen = None
    test_gen = None

    if train:
        trn_all_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                          channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                          horizontal_flip=True)
        trn_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                          channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                          horizontal_flip=True)
        val_datagen = ImageDataGenerator()
        if X != None and y != None:
            trn_all_gen = trn_all_datagen.flow(X, y, batch_size=batch_size,
                                                            shuffle=True)
        else:
            trn_all_gen = trn_all_datagen.flow_from_directory(trn_all_path, target_size=size, batch_size=batch_size,
                                                            class_mode='categorical', shuffle=True)
        if X_trn != None and y_trn!= None:
            trn_gen = trn_datagen.flow(X_trn, y_trn, batch_size=batch_size,
                                                            shuffle=True)
        else:
            trn_gen = trn_datagen.flow_from_directory(trn_path, target_size=size, batch_size=batch_size,
                                                            class_mode='categorical', shuffle=True)
        if X_val != None and y_val!= None:
            val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size,
                                                               shuffle=True)
        else:
            val_gen = val_datagen.flow_from_directory(val_path, target_size=size, batch_size=batch_size,
                                                               class_mode='categorical', shuffle=True)
    
    if test:
        # test_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                          channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                          horizontal_flip=True)
        test_gen = test_datagen.flow_from_directory(test_path, target_size=size, batch_size=batch_size,
                                                        class_mode='categorical', shuffle=False)
    return trn_all_gen, trn_gen, val_gen, test_gen


def inception():
    def inception_preprocess(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    size = inception_size
    n_classes = 8
    lr = 0.001

    img_input = Input(shape=(3, size[0], size[1]))
    inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=img_input)

    for layer in inception.layers:
        layer.trainable = False

    output = inception.output
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(n_classes, activation='softmax', name='predictions')(output)

    model = Model(inception.input, output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=True),
                  metrics=["accuracy"])
    return model

def get_inception_gens(train=True, test=True):
    trn_all_path = 'train_all/'
    trn_path = 'train/'
    val_path = 'valid/'
    test_path = 'test/'

    size = inception_size
    batch_size = 64

    trn_all_gen = None
    trn_gen = None
    val_gen = None
    test_gen = None

    if train:
        trn_all_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                              channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                              horizontal_flip=True, rescale=2./255, samplewise_center=True)
        trn_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                              channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                              horizontal_flip=True, rescale=2./255, samplewise_center=True)
        val_datagen = ImageDataGenerator(rescale=2./255, samplewise_center=True)
        trn_all_gen = trn_all_datagen.flow_from_directory(trn_all_path, target_size=size, batch_size=batch_size,
                                                            class_mode='categorical', shuffle=True)
        trn_gen = trn_datagen.flow_from_directory(trn_path, target_size=size, batch_size=batch_size,
                                                            class_mode='categorical', shuffle=True)
        val_gen = val_datagen.flow_from_directory(val_path, target_size=size, batch_size=batch_size,
                                                               class_mode='categorical', shuffle=True)
    if test:
        # test_datagen = ImageDataGenerator(rescale=2./255, samplewise_center=True)
        test_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                              channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                              horizontal_flip=True, rescale=2./255, samplewise_center=True)
        test_gen = test_datagen.flow_from_directory(test_path, target_size=size, batch_size=batch_size,
                                                        class_mode=None, shuffle=False)
    return trn_all_gen, trn_gen, val_gen, test_gen




if __name__ == '__main__':
    X, X_trn, X_val, y, y_trn, y_val = load_data(saved=True)

    # model = inception()
    # trn_all_gen, trn_gen, val_gen, test_gen = get_inception_gens()
    model = vgg16()
    trn_all_gen, trn_gen, val_gen, test_gen = get_vgg_gens(X=X, X_trn=X_trn, X_val=X_val, y=y, y_trn=y_trn, y_val=y_val)


    nb_epoch = 10

    nb_runs = 5
    nb_aug = 5

    # model.fit_generator(trn_all_gen, samples_per_epoch=nb_trn_all_samples, nb_epoch=nb_epoch, verbose=2)
    # model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                # validation_data=val_gen, nb_val_samples=nb_val_samples)
    # model.save_weights('weights/vgg16_only_fish_10epochs_relabeled.h5')
    model.load_weights('weights/vgg16_all_10epochs_relabeled.h5')
    image_paths = [
        "train_all/ALB/img_00010.jpg"
    ]
    
    # ACTIVATION MAP STUFF

    def get_output_layer(model, layer_name):
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer

    original_img = cv2.imread("train_all/ALB/img_00010.jpg", 1)
    original_img = cv2.resize(original_img, (vgg_size[1], vgg_size[0]), cv2.INTER_LINEAR)
    width, height, _ = original_img.shape

    #Reshape to the network input shape (3, w, h).
    img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])
    
    #Get the 512 input weights to the softmax.
    print model.layers[-1].get_weights()[0].shape
    print model.layers[-2].get_weights()
    print model.layers[-3].get_weights()[0].shape
    print model.layers[-4].get_weights()[0].shape
    print model.layers[-5].get_weights()
    print model.layers[-6].get_weights()[0].shape
    print model.layers[-7].get_weights()[0].shape
    print model.layers[-8].get_weights()
    print model.layers[-9].get_weights()
    print model.layers[-10].get_weights()[0].shape
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "conv5_3")
    get_output = K.function([model.layers[0].input, K.learning_phase()], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img, 1])
    print conv_outputs.shape
    conv_outputs = conv_outputs[0, :, :, :]

    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
    # print class_weights
    print conv_outputs.shape
    print class_weights.shape
    print len(class_weights[:, 1])
    print "predictions", predictions
    for i, w in enumerate(class_weights[:, 1]):
        cam += w * conv_outputs[i, :, :]
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap*0.5 + original_img
    cv2.imwrite("hmm.jpg", img)

    # ACTIVATION MAP STUFF

    exit(1)

    start_time = time.time()
    predictions_full = np.zeros((nb_test_samples, nb_classes))
    
    for run in range(nb_runs):
        print("\nStarting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
        predictions_aug = np.zeros((nb_test_samples, nb_classes))
        
        for aug in range(nb_aug):
            print("\n--Predicting on Augmentation {0} of {1}...\n".format(aug+1, nb_aug))
            model = vgg16()
            model.load_weights('weights/vgg16_only_fish_10epochs_relabeled.h5')
            trn_all_gen, trn_gen, val_gen, test_gen = get_vgg_gens(train=False)
            predictions_aug += model.predict_generator(test_gen, val_samples=nb_test_samples)

        predictions_aug /= nb_aug
        predictions_full += predictions_aug

        print '{} runs in {} sec'.format(run+1, time.time() - start_time)
    
    predictions_full /= nb_runs
    np.save("pred/pred_vgg16_only_fish_10epochs_relabeled.npy", predictions_full)
    exit(1)

    predictions_full = np.load("pred/pred_vgg16_all_10epochs_relabeled.npy")
    pred_no_fish = np.load("pred/pred_vgg16_no_fish_10epochs.npy")
    preds = np.clip(predictions_full,0.02, 1.00, out=None)
    # preds = np.zeros((nb_test_samples, nb_classes)) + .0414
    # for i in range(predictions_full.shape[0]):
    #     preds[i][np.argmax(predictions_full[i])] = .71


    with open('submissions/submission11.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(test_gen.filenames):
            pred = ['%.6f' % (p/np.sum(preds[i, :])) for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")