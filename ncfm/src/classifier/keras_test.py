import cv2
import numpy as np
import os
import time

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

import sys
sys.path.append('/home/mzhao/Desktop/kaggle/ncfm/src')
from lib import models
from lib import utils

valid_percent = .15

vgg_size = (270, 480)

fish_types = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
# fish_counts = [1745,202,117,68,442,286,177,740]
# fish_types = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
# fish_counts = [1745,202,117,68,286,177,740]

# fish_cumulative_counts = [0] + [sum(fish_counts[:i+1]) for i in range(len(fish_counts))]
# nb_trn_all_samples = fish_cumulative_counts[-1]
# nb_trn_samples = int(sum([((1 - valid_percent)*100*c)//100 for c in fish_counts]))
# nb_val_samples = nb_trn_all_samples - nb_trn_samples
# print [((1 - valid_percent)*100*c)//100 for c in fish_counts]
# print fish_cumulative_counts
# print nb_trn_all_samples, nb_trn_samples, nb_val_samples

nb_test_samples = 1000
nb_classes = len(fish_types)

k=6

if __name__ == '__main__':
    X, y, X_folds, y_folds, filename_folds = utils.load_data(fish_types=fish_types, size=vgg_size,
                                        saved=True, savefileX='X.npy', savefileY='y.npy', k=k)
    # exit(1)
    X_trn = []
    X_val = []
    y_trn = []
    y_val = []
    for j in range(k):
        if j == 0:
            X_val += list(X_folds[j])
            y_val += list(y_folds[j])
        else:
            X_trn += list(X_folds[j])
            y_trn += list(y_folds[j])
    X_trn = np.array(X_trn)
    X_val = np.array(X_val)
    y_trn = np.array(y_trn)
    y_val = np.array(y_val)

    model = models.vgg16()
    # trn_all_gen = models.get_train_all_gens(X, y, size=vgg_size, batch_size=16)
    trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, size=vgg_size, batch_size=16)
    # test_gen = models.get_test_gens(size=vgg_size, batch_size=16)

    nb_epoch = 10

    nb_runs = 5
    nb_aug = 5

    print "start training"

    # models.train_all(model, trn_all_gen, nb_trn_all_samples=nb_trn_all_samples,
                     # nb_epoch=nb_epoch, weightfile='vgg16_10epochs_relabeled.h5')
    models.train_val(model, trn_gen, val_gen, nb_trn_samples=X_trn.shape[0], nb_val_samples=X_val.shape[0],
                    nb_epoch=nb_epoch, weightfile='vgg16_stack.h5')
    exit(1)


    # model.load_weights('weights/train_all/vgg16_10epochs_relabeled.h5')
    model.load_weights('weights/train_val/vgg16_stack.h5')

    models.predict(model, predfile='pred_vgg16_yolo_10epochs_relabeled.npy',
                   nb_test_samples=1000, nb_classes=8, nb_runs=1, nb_aug=1)

    exit(1)

    # utils.write_submission(test_gen.filenames, predfile='pred_vgg16_yolo_10epochs_relabeled.npy', subfile='submission14.csv')
    







