import cv2
import models
import numpy as np
import os
import time
import utils

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

fish_types = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
fish_counts = [1745,202,117,68,442,286,177,740]
# fish_types = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
# fish_counts = [1745,202,117,68,286,177,740]

fish_cumulative_counts = [0] + [sum(fish_counts[:i+1]) for i in range(len(fish_counts))]
nb_trn_all_samples = fish_cumulative_counts[-1]
nb_trn_samples = int(sum([((1 - valid_percent)*100*c)//100 for c in fish_counts]))
nb_val_samples = nb_trn_all_samples - nb_trn_samples
# print fish_cumulative_counts
# print nb_trn_all_samples, nb_trn_samples, nb_val_samples

nb_test_samples = 1000
nb_classes = len(fish_types)


if __name__ == '__main__':
    X, X_trn, X_val, y, y_trn, y_val = utils.load_data(valid_percent=valid_percent, fish_types=fish_types, fish_counts=fish_counts, saved=True)

    model = models.vgg16()
    trn_all_gen = models.get_train_all_gens(X, y, size=vgg_size, batch_size=16)
    trn_gen, val_gen = get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, size=vgg_size, batch_size=16)
    test_gen = models.get_test_gens(size=vgg_size, batch_size=16)

    nb_epoch = 10

    nb_runs = 5
    nb_aug = 5

    # models.train_all(model, trn_all_gen, nb_trn_all_samples=nb_trn_all_samples, nb_epoch=nb_epoch, weightfile='vgg16_10epochs_relabeled.h5')
    models.train_val(model, trn_gen, val_gen, nb_trn_samples=nb_trn_samples, nb_val_samples=nb_val_samples, nb_epoch=nb_epoch, weightfile='vgg16_10epochs_relabeled.h5')
    exit(1)

    model.load_weights('weights/vgg16_10epochs_relabeled.h5')
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
    cam = np.zeros(dtype = np.float32, shape=conv_outputs.shape[1:3])
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

    model.load_weights('weights/train_all/vgg16_only_fish_10epochs_relabeled.h5')
    models.predict(model, predfile='pred_vgg16_only_fish_10epochs_relabeled.npy')
    exit(1)

    utils.write_submission(predfile='pred_vgg16_only_fish_10epochs_relabeled.npy', subfile='submission12.csv')
    







