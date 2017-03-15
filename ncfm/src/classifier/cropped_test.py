import cv2
import numpy as np
import os
import sklearn
import time

import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)

import theano

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

import sys
sys.path.append('/home/mzhao/Desktop/kaggle/ncfm/src')
from lib import models
from lib import utils


# vgg_size = (300,300)
vgg_size = (270,480)

fish_types = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
# fish_counts = [1745,202,117,68,442,286,177,740]
# fish_multipliers = [1,1,1,1,1,1,1,1]
# fish_multipliers = [1,2,2,1,1,1,1,1]
# fish_multipliers = [1,3,5,2,2,2,2,1]

# fish_types = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
# fish_counts = [1722,199,115,68,286,175,732]
# fish_counts = [1745,202,117,68,286,177,740]
# fish_multipliers = [1,1,1,1,1,1,1]

# fish_cumulative_counts = [0] + [sum(fish_counts[:i+1]) for i in range(len(fish_counts))]
# nb_trn_all_samples = fish_cumulative_counts[-1]
# trn_samples_counts = [((1 - valid_percent)*100*c)//100 for c in fish_counts]
# nb_val_samples = nb_trn_all_samples - int(sum(trn_samples_counts))
# nb_trn_samples = int(sum([trn_samples_counts[i] * fish_multipliers[i] for i in range(len(fish_multipliers))]))
# print fish_cumulative_counts
# print nb_trn_all_samples, nb_trn_samples, nb_val_samples

nb_test_samples = 1000
nb_classes = len(fish_types)

k = 3

if __name__ == '__main__':
    X, y, X_folds, y_folds, filename_folds = utils.load_data_cropped(fish_types=fish_types, size=vgg_size,
                                                       # saved=True, savefileX='X_cropped_borderless_270x480.npy', savefileY='y_cropped_borderless_270x480.npy')
                                                       # saved=True, savefileX='X_cropped2_borderless_270x480.npy', savefileY='y_cropped2_borderless_270x480.npy')
                                                       saved=True, savefileX='X_all_cropped2_borderless_270x480.npy', savefileY='y_all_cropped2_borderless_270x480.npy', k=k)
    
    X_trn = []
    X_val = []
    y_trn = []
    y_val = []
    for j in range(k):
        if j == 0:
            # X_val += list(X_folds[j])
            # y_val += list(y_folds[j])
            X_trn += list(X_folds[j])
            y_trn += list(y_folds[j])
        else:
            X_trn += list(X_folds[j])
            y_trn += list(y_folds[j])
    X_trn = np.array(X_trn)
    X_val = np.array(X_val)
    y_trn = np.array(y_trn)
    y_val = np.array(y_val)

    # print np.sum(y_trn, axis=0)
    # print np.sum(y_val, axis=0)
    # exit(1)
    # trn_mean = np.average(np.average(np.average(X_trn, axis=0), axis=2), axis=1).astype('uint8')
    # val_mean = np.average(np.average(np.average(X_val, axis=0), axis=0), axis=0)
    # for i in range(X_trn.shape[0]):
    #     for j in range(X_trn.shape[2]):
    #         for k in range(X_trn.shape[3]):
    #             X_trn[i,:,j,k] -= trn_mean
    # print trn_mean
    # print X_trn.shape
    # X_trn = X_trn.transpose((0, 2, 3, 1))
    # print np.average(np.average(np.average(X_trn, axis=0), axis=0), axis=0).astype('uint8')
    # print X_trn.shape
    # X_trn -= trn_mean
    # print np.average(np.average(np.average(X_trn, axis=0), axis=0), axis=0).astype('uint8')
    # X_trn = X_trn.transpose((0, 3, 1, 2))
    # print X_trn.shape
    # X_val -= val_mean
    # print trn_mean, val_mean
    # print np.average(np.average(np.average(X_trn, axis=0), axis=2), axis=1)
    # TRY USING THESE MEANS, TRY A DIFFERENT LEARNING RATE ADAPTER (optimizer), TRY NOT FLIPPING 2ND COLUMN
    # exit(1)
    batch_size = 16

    model = models.vgg16(size=vgg_size, nb_classes=len(fish_types))
    trn_all_gen = models.get_train_all_gens(X_trn, y_trn, size=vgg_size, batch_size=batch_size)
    # trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, size=vgg_size, batch_size=batch_size)
    # test_gen = models.get_test_gens(size=vgg_size, batch_size=16, test_path='test_cropped/')

    nb_epoch = 12

    nb_runs = 5
    nb_aug = 5

    # exit(1)

    # model.load_weights('weights/train_val/vgg16_cropped.h5')
    # pred = model.predict(X_val, batch_size=16)
    # actual_pred = np.zeros(pred.shape[0])
    # actual_label = np.zeros(pred.shape[0])
    # for i in range(pred.shape[0]):
    #     actual_pred[i] = np.argmax(pred[i])
    #     actual_label[i] = np.argmax(y_val[i])
    # conf = sklearn.metrics.confusion_matrix(actual_pred, actual_label)
    # print conf
    # print float(np.trace(conf))/float(np.sum(conf))

    # exit(1)


    # checkpointer = ModelCheckpoint(filepath="/home/mzhao/Desktop/kaggle/ncfm/yolo"+
    #                                         "/localized_weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
    #                                 verbose=1, save_weights_only=True, save_best_only=True)
    # model.fit_generator(trn_gen, samples_per_epoch=X_trn.shape[0], nb_epoch=nb_epoch, verbose=2,
            # validation_data=val_gen, nb_val_samples=X_val.shape[0], callbacks=[checkpointer])
    checkpointer = ModelCheckpoint(filepath="/home/mzhao/Desktop/kaggle/ncfm/yolo2"+
                                            "/localized_weights.{epoch:02d}.hdf5", 
                                    verbose=1, save_weights_only=True)
    model.fit_generator(trn_all_gen, samples_per_epoch=X_trn.shape[0], nb_epoch=nb_epoch, verbose=2,
            callbacks=[checkpointer])

    # models.train_all(model, trn_all_gen, nb_trn_all_samples=nb_trn_all_samples,
    #                  nb_epoch=nb_epoch, weightfile='vgg16_cropped2.h5')
    # models.train_val(model, trn_gen, val_gen, nb_trn_samples=nb_trn_samples, nb_val_samples=nb_val_samples,
    #                  nb_epoch=nb_epoch, weightfile='vgg16_cropped2.h5')
    exit(1)

    model.load_weights("/home/mzhao/Desktop/kaggle/ncfm/yolo2/localized_weights.13-0.32.hdf5")

    best = np.load("pred/pred_vgg16_cropped22.npy")
    p2 = np.load("pred/pred_vgg16_all_10epochs_relabeled.npy")
    for i in range(len(best)):
        best[i] *= (1 - p2[i][4])
    best = np.insert(best, 4, p2[:,4], axis=1)
    best += p2
    best /= 2

    # best = np.load("pred/pred_vgg16_cropped22.npy")

    X_test = []
    index = 0
    for file in sorted(os.listdir("test_cropped2/unknown")):
        # print file
        path = "test_cropped2/unknown/{}".format(file)
        # img = np.array(keras.preprocessing.image.load_img(path, target_size=vgg_size))
        # img = skimage.io.imread(path)
        img = plt.imread(path)
        if img.shape[0] > img.shape[1]:
            img = img.transpose((1, 0, 2))
        # plt.imshow(img)
        # plt.show()
        # avg = np.mean(np.mean(img, axis=0), axis=0)
        # if img.shape[0] > img.shape[1]:
        #     temp = np.zeros((img.shape[0], img.shape[0], 3)) + avg
        #     start_index = img.shape[0] / 2 - img.shape[1]/2
        #     temp[:,start_index:start_index+img.shape[1],:] = img
        # elif img.shape[0] < img.shape[1]:
        #     temp = np.zeros((img.shape[1], img.shape[1], 3)) + avg
        #     start_index = img.shape[1] / 2 - img.shape[0]/2
        #     temp[start_index:start_index+img.shape[0],:,:] = img
        # # plt.imshow(temp)
        # # plt.show()
        # img = temp
        # # img = skimage.transform.resize(img, vgg_size).transpose((2, 0, 1))
        # # print img.shape
        img = cv2.resize(img, (vgg_size[1], vgg_size[0]), cv2.INTER_LINEAR)
        # plt.imshow(img)
        # plt.show()
        # exit(1)
        # print img.shape
        img = img.transpose((2, 0, 1))
        # print img.shape
        X_test += [img]
    X_test = np.array(X_test)
    print X_test.shape
    # exit(1)

    X_test = []
    for file in sorted(os.listdir("test_cropped2/unknown")):
        path = "test_cropped2/unknown/{}".format(file)
        img = cv2.imread(path)
        img = cv2.resize(img, (vgg_size[1], vgg_size[0]), cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        X_test += [img]
    X_test = np.array(X_test)

    predictions_fold = np.zeros(best.shape)
    for run in range(nb_runs):
        print("Starting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
        test_gen = models.get_test_gens(X_test=X_test, size=vgg_size, batch_size=16)
        predictions_run = model.predict_generator(test_gen, val_samples=nb_test_samples)
        predictions_fold += predictions_run
        print np.sum(predictions_run, axis=0)

        actual_pred = np.zeros(predictions_run.shape[0])
        actual_label = np.zeros(predictions_run.shape[0])
        for ii in range(best.shape[0]):
            actual_pred[ii] = np.argmax(best[ii])
            actual_label[ii] = np.argmax(predictions_run[ii])
        conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
        print conf
        print float(np.trace(conf))/float(np.sum(conf))

    predictions_fold /= nb_runs
    np.save("/home/mzhao/Desktop/kaggle/ncfm/yolo/localized_pred.npy", predictions_fold)

    exit(1)



    # legit = np.loadtxt('submissions/e2e_cropped_avg_blend.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6,7,8))
    # preds = np.loadtxt('submissions/better_crops.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6,7,8))
    # preds2 = np.loadtxt('submissions/better_crops2.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6,7,8))
    
    # print np.sum(legit, axis=0)
    # print np.sum(preds, axis=0)
    # print np.sum(preds2, axis=0)

    # actual_pred = np.zeros(preds.shape[0])
    # actual_label = np.zeros(preds.shape[0])
    # for i in range(preds.shape[0]):
    #     actual_pred[i] = np.argmax(preds2[i])
    #     actual_label[i] = np.argmax(legit[i])
    # conf = sklearn.metrics.confusion_matrix(actual_pred, actual_label)
    # print conf
    # print float(np.trace(conf))/float(np.sum(conf))

    # # print actual_pred
    # # print actual_label
    # exit(1)

    preds = np.load("pred/pred_vgg16_cropped22.npy")
    print np.sum(preds, axis=0)
    # exit(1)

    # p2 = np.load("pred/ssd_blend.npy")
    p2 = np.load("pred/pred_vgg16_all_10epochs_relabeled.npy")
    for i in range(len(preds)):
        preds[i] *= (1 - p2[i][4])
    preds = np.insert(preds, 4, p2[:,4], axis=1)
    preds += p2
    preds /= 2

    print np.sum(p2, axis=0)
    print np.sum(preds, axis=0)

    actual_pred = np.zeros(preds.shape[0])
    actual_label = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        actual_pred[i] = np.argmax(preds[i])
        actual_label[i] = np.argmax(p2[i])
    conf = sklearn.metrics.confusion_matrix(actual_pred, actual_label)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))

    print actual_pred
    print actual_label

    # exit(1)


    preds = np.clip(preds, 0.02, .98, out=None)

    filenames = sorted(os.listdir("test_cropped/unknown"))
    # print filenames[:10]
    with open('submissions/better_crops3.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % (p/np.sum(preds[i, :])) for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")

    # utils.write_submission(test_gen.filenames, predfile='pred_vgg16_cropped.npy', subfile='submission13.csv')
    







