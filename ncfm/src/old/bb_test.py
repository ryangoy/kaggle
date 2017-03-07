import cv2
import matplotlib.pyplot as plt
import models
import numpy as np
import os
import sklearn
import time
import utils

seed = 0
np.random.seed(seed)

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

# fish_types = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
# fish_counts = [1745,202,117,68,442,286,177,740]
# fish_multipliers = [1,1,1,1,1,1,1,1]
# fish_multipliers = [1,2,2,1,1,1,1,1]
# fish_multipliers = [1,3,5,2,2,2,2,1]
fish_types = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
fish_counts = [1720,198,117,68,286,174,732]
fish_multipliers = [1,1,1,1,1,1,1]

fish_cumulative_counts = [0] + [sum(fish_counts[:i+1]) for i in range(len(fish_counts))]
nb_trn_all_samples = fish_cumulative_counts[-1]
trn_samples_counts = [((1 - valid_percent)*100*c)//100 for c in fish_counts]
nb_val_samples = nb_trn_all_samples - int(sum(trn_samples_counts))
nb_trn_samples = int(sum([trn_samples_counts[i] * fish_multipliers[i] for i in range(len(fish_multipliers))]))
# print fish_cumulative_counts
# print nb_trn_all_samples, nb_trn_samples, nb_val_samples

nb_test_samples = 1000
nb_classes = len(fish_types)


if __name__ == '__main__':
    X, X_trn, X_val, y, y_trn, y_val = utils.load_data_bbox_2point(valid_percent=valid_percent, fish_types=fish_types, fish_counts=fish_counts, 
                                                       fish_multipliers=fish_multipliers, size=vgg_size,
                                                       # saved=True, savefileX='X_preprocessed.npy', savefileY='y_preprocessed.npy')
                                                       # saved=True, savefileX='X_reg.npy', savefileY='y_reg.npy', output='regression')
                                                       saved=True, savefileX='X_cls.npy', savefileY='y_cls.npy', output='classification')
    model = models.vgg16_bb(output='classification')
    trn_all_gen = models.get_train_all_gens(X, y, size=vgg_size, batch_size=16)
    trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, size=vgg_size, batch_size=16)
    test_gen = models.get_test_gens(size=vgg_size, batch_size=16)

    nb_epoch = 10

    nb_runs = 5
    nb_aug = 5


    # model.load_weights('weights/train_val/bb/vgg16_cls_10.h5')
    # pred = model.predict(X_val, batch_size=16)
    # for i in range(len(X_val)):
    #     plt.figure()
    #     # plt.plot(pred[i])
    #     plt.imshow(X[i].transpose(1, 2, 0))
    #     # x1c = pred[i][0]
    #     # y1c = pred[i][1]
    #     # x2c = pred[i][2]
    #     # y2c = pred[i][3]
    #     x1c = np.argmax(pred[i][:480])
    #     y1c = np.argmax(pred[i][480:750])
    #     x2c = np.argmax(pred[i][750:1230])
    #     y2c = np.argmax(pred[i][1230:])
    #     plt.scatter(x1c, y1c, s=25, c='green', marker='o')
    #     plt.scatter(x2c, y2c, s=25, c='red', marker='o')
    #     plt.show()

    # actual_pred = np.zeros(pred.shape[0])
    # actual_label = np.zeros(pred.shape[0])
    # for i in range(pred.shape[0]):
    #     actual_pred[i] = np.argmax(pred[i])
    #     actual_label[i] = np.argmax(y_val[i])
    # conf = sklearn.metrics.confusion_matrix(actual_pred, actual_label)
    # print conf
    # print float(np.trace(conf))/float(np.sum(conf))

    # models.train_all(model, trn_all_gen, nb_trn_all_samples=nb_trn_all_samples,
    #                  nb_epoch=nb_epoch, weightfile='bb/vgg16_20epochs.h5')
    models.train_val(model, trn_gen, val_gen, nb_trn_samples=nb_trn_samples, nb_val_samples=nb_val_samples,
                     nb_epoch=nb_epoch, weightfile='bb/vgg16_cls_10.h5')
    exit(1)


    model.load_weights('weights/train_val/vgg16_mult2_10epochs_relabeled.h5')
    models.predict(model, predfile='pred_vgg16_mult2_10epochs_relabeled.npy',
                   nb_test_samples=1000, nb_classes=8, nb_runs=5, nb_aug=5)
    # exit(1)







