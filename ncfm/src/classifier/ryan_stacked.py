import cv2
import numpy as np
import sklearn
import os
import time
import sys
import pandas as pd
import pickle
from operator import itemgetter
from os.path import join
import xgboost as xgb
from keras.callbacks import ModelCheckpoint
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
sys.path.append('/home/ryan/cs/kaggle/ncfm/src')
from lib import models
from lib import utils

vgg_size = (224, 224)
fish_types = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
nb_test_samples = 1000
nb_classes = len(fish_types)
nb_epoch_e2e = 20
nb_epoch_localized = 20
nb_runs = 5
batch_size = 16
seed = 0
np.random.seed(seed)
val_size = .2
stacked_val_size = .1

VGG_weights = '/home/ryan/cs/datasets/ncfm/weights/vgg16_full.h5'
weights_path = '/home/ryan/cs/datasets/ncfm/stacked/output_weights'
images_path = '/home/ryan/cs/datasets/ncfm/train_categories'
cropped_images_path = '/home/ryan/cs/datasets/ncfm/train_categories_cropped'
e2e_preloaded_images = '/home/ryan/cs/datasets/ncfm/stacked/X_e2e.npy'
loc_preloaded_images = '/home/ryan/cs/datasets/ncfm/stacked/X_loc.npy'
e2e_preloaded_labels = '/home/ryan/cs/datasets/ncfm/stacked/one_hot_labels_e2e.npy'
loc_preloaded_labels = '/home/ryan/cs/datasets/ncfm/stacked/one_hot_labels_loc.npy'
bboxes_path = '/home/ryan/cs/datasets/ncfm/stacked/predictions_no_fish.pkl'

# change these to None if you want to train
# best_e2e_weights = join(weights_path, 'e2e_weights.05-0.43.hdf5')
# best_loc_weights = join(weights_path, 'loc_weights.11-0.30.hdf5')
best_e2e_weights = None
best_loc_weights = None

X_trn = None
X_val = None
y_trn = None
y_val = None
X_sval= None
y_sval= None
trn_gen = None
val_gen = None

def run():

    e2e_load_data(images_path)
    e2e_pred = e2e_run(name="e2e", weights = best_e2e_weights)
    loc_load_data(cropped_images_path)
    loc_pred = e2e_run(name="loc", weights = best_loc_weights)


    if best_e2e_weights is None or best_loc_weights is None:
        return
    df1 = pd.DataFrame(e2e_pred)
    df2 = pd.DataFrame(loc_pred)
    df2.columns = [str(col) + '_loc' for col in df2.columns]
    
    stacked_input = pd.concat([df1, df2], axis=1)

    train_stacked(stacked_input, y_val)


def e2e_load_data(images_path):

    if os.path.isfile(e2e_preloaded_images) and os.path.isfile(e2e_preloaded_labels) :
        X = np.load(e2e_preloaded_images)
        y = np.load(e2e_preloaded_labels)
    else:
        print "[LOADING DATA]"
        X = []
        y = []
        index = 0
        paths = []
        for i in range(len(fish_types)):
            fish = fish_types[i]
            # for file in os.listdir("preprocessed_train/{}".format(fish)):
            for file in sorted(os.listdir(join(images_path, fish))):
                # path = "preprocessed_train/{}/{}".format(fish, file)
                path = join(images_path, fish, file)
                paths += [(file, path, i)]
        paths = sorted(paths, key=itemgetter(0))
        for p in paths:
            # img = np.array(keras.preprocessing.image.load_img(path, target_size=vgg_size))
            # img = skimage.io.imread(path)
            img = cv2.imread(p[1])
            # img = skimage.transform.resize(img, vgg_size).transpose((2, 0, 1))
            # print img.shape

            img = cv2.resize(img, vgg_size, cv2.INTER_LINEAR)

            label = [0 for _ in range(len(fish_types))]
            label[p[2]] = 1
            X += [img]
            y += [label]
        X = np.array(X)
        y = np.array(y)
        np.save(e2e_preloaded_images, X)
        np.save(e2e_preloaded_labels, y)

   
    permute = np.random.permutation(X.shape[0])
    X_shuf = X[permute]
    y_shuf = y[permute]
    split_index = int(X.shape[0]*(1-val_size-stacked_val_size))
    second_split = int(X.shape[0]*(1-stacked_val_size))
    global X_trn
    global X_val
    global y_trn
    global y_val
    global X_sval
    global y_sval
    global trn_gen
    global val_gen
    X_trn = X[:split_index]
    X_val = X[split_index:second_split]
    y_trn = y[:split_index]
    y_val = y[split_index:second_split]
    X_sval = X[second_split:]
    y_sval = y[second_split:]
    trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, 
                                                    size=vgg_size, batch_size=batch_size)


def loc_load_data(images_path):

    if os.path.isfile(loc_preloaded_images) and os.path.isfile(loc_preloaded_labels) :
        X = np.load(loc_preloaded_images)
        y = np.load(loc_preloaded_labels)
    else:
        print "[LOADING DATA]"
        X = []
        y = []
        index = 0
        paths = []
        for i in range(len(fish_types)):
            fish = fish_types[i]
            # for file in os.listdir("preprocessed_train/{}".format(fish)):
            for file in sorted(os.listdir(join(images_path, fish))):
                # path = "preprocessed_train/{}/{}".format(fish, file)
                path = join(images_path, fish, file)
                paths += [(file, path, i)]
        paths = sorted(paths, key=itemgetter(0))
        for p in paths:
            # img = np.array(keras.preprocessing.image.load_img(path, target_size=vgg_size))
            # img = skimage.io.imread(path)
            img = cv2.imread(p[1])
            # img = skimage.transform.resize(img, vgg_size).transpose((2, 0, 1))
            # print img.shape

            img = cv2.resize(img, vgg_size, cv2.INTER_LINEAR)

            label = [0 for _ in range(len(fish_types))]
            label[p[2]] = 1
            X += [img]
            y += [label]
        X = np.array(X)
        y = np.array(y)
        np.save(loc_preloaded_images, X)
        np.save(loc_preloaded_labels, y)

   
    permute = np.random.permutation(X.shape[0])
    X_shuf = X[permute]
    y_shuf = y[permute]
    split_index = int(X.shape[0]*(1-val_size))
    global X_trn
    global X_val
    global y_trn
    global y_val
    global trn_gen
    global val_gen
    X_trn = X[:split_index]
    X_val = X[split_index:]
    y_trn = y[:split_index]
    y_val = y[split_index:]
    trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, 
                                                    size=vgg_size, batch_size=batch_size)

def e2e_run(name, nb_epoch=nb_epoch_e2e, batch_size=batch_size, weights=None):
    print '[STARTING END TO END RUN]'
    if weights is not None:
        print "[LOADING END TO END MODEL]"
        model = models.VGG16_test(weights)
        pred = model.predict_generator(val_gen, val_samples=X_val.shape[0])
        print_confusion_matrix(y_val, pred)
        return pred
        
    else:
        print "[TRAINING END TO END MODEL]"
        model = models.VGG16(VGG_weights, input_shape=vgg_size+(3,), classes=nb_classes)
        


        checkpointer = ModelCheckpoint(filepath=join(weights_path, name+"_weights.{epoch:02d}-{val_loss:.2f}.hdf5"), 
                                        verbose=1, save_weights_only=True)
        model.fit_generator(trn_gen, samples_per_epoch=X_trn.shape[0], nb_epoch=nb_epoch, verbose=1,
                    validation_data=val_gen, nb_val_samples=X_val.shape[0], callbacks=[checkpointer])


        print "[FINISHED TRAINING END TO END MODEL]"

def print_confusion_matrix(label, pred):
    actual_pred = np.zeros(pred.shape[0])
    actual_label = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        actual_pred[i] = np.argmax(pred[i])
        actual_label[i] = np.argmax(label[i])
    conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))

def train_stacked(train, labels):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 8
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = 0
    num_rounds = 1000
    plst = list(param.items())
    xg_trn = xgb.DMatrix(train, label=np.argmax(labels, axis=1))
    xg_val = xgb.DMatrix(train, label=np.argmax(labels, axis=1))


    watchlist = [ (xg_trn,'train'), (xg_val, 'val') ]
    model = xgb.train(plst, xg_trn, num_rounds, watchlist, early_stopping_rounds=20)


    #pred = model.predict(xg_val)



if __name__ == "__main__":
    run()