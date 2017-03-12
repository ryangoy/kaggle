import cv2
import numpy as np
import os
import sklearn
import time

seed = 0
np.random.seed(seed)

from keras.callbacks import ModelCheckpoint

import sys
sys.path.append('/home/mzhao/Desktop/kaggle/ncfm/src')
from lib import models
from lib import utils

vgg_size = (270, 480)

fish_types = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

nb_test_samples = 1000
nb_classes = len(fish_types)

k = 3

nb_epoch_e2e = 15
nb_epoch_localized = 15

nb_runs = 5

def e2e_run(nb_epoch, train=False, pred_val=False, early_weights=True):
    X, y, X_folds, y_folds, filename_folds = utils.load_data(fold_file='/home/mzhao/Desktop/kaggle/ncfm/runs/run1/folds.json',
                                        fish_types=fish_types, size=vgg_size,
                                        saved=True, savefileX='X.npy', savefileY='y.npy', k=3)
    # return

    if train:
        for i in range(k):
            # if i == 0 or i == 1:
            #     continue
            start_time = time.time()
            X_trn = []
            X_val = []
            y_trn = []
            y_val = []
            for j in range(k):
                if j == i:
                    X_val += list(X_folds[j])
                    y_val += list(y_folds[j])
                else:
                    X_trn += list(X_folds[j])
                    y_trn += list(y_folds[j])
            X_trn = np.array(X_trn)
            X_val = np.array(X_val)
            y_trn = np.array(y_trn)
            y_val = np.array(y_val)

            # print X_trn.shape, X_val.shape, y_trn.shape, y_val.shape

            model = models.vgg16()
            trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, size=vgg_size, batch_size=16)

            print "[START] training e2e fold {}/{}".format(i+1,k)

            checkpointer = ModelCheckpoint(filepath="/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold{}".format(i+1)+
                                                    "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                            verbose=1, save_weights_only=True)
            model.fit_generator(trn_gen, samples_per_epoch=X_trn.shape[0], nb_epoch=nb_epoch, verbose=2,
                    validation_data=val_gen, nb_val_samples=X_val.shape[0], callbacks=[checkpointer])

            print "[FINISHED] training e2e fold {}/{} in {} sec".format(i+1,k,time.time()-start_time)
            return
        # return

    weights = [["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold1/weights.06-0.39.hdf5",
                "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold1/weights.14-0.35.hdf5"],
               ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold2/weights.07-0.27.hdf5",
                "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold2/weights.11-0.26.hdf5"],
               ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold3/weights.12-0.19.hdf5"]]

    if pred_val:
        # preds = []
        for i in range(k):
            start_time = time.time()
            X_trn = []
            X_val = []
            y_trn = []
            y_val = []
            for j in range(k):
                if j == i:
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
                
            if early_weights:
                model.load_weights(weights[i][0])
            else:
                model.load_weights(weights[i][0])

            predictions_fold = np.zeros((X_val.shape[0], nb_classes))
            for run in range(nb_runs):
                print("Starting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
                trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, size=vgg_size, batch_size=16)
                predictions_run = model.predict_generator(val_gen, val_samples=X_val.shape[0])
                predictions_fold += predictions_run
                print np.sum(y_val, axis=0)
                print np.sum(predictions_run, axis=0)

                actual_pred = np.zeros(y_val.shape[0])
                actual_label = np.zeros(predictions_fold.shape[0])
                for ii in range(y_val.shape[0]):
                    actual_pred[ii] = np.argmax(y_val[ii])
                    actual_label[ii] = np.argmax(predictions_run[ii])
                conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
                print conf
                print float(np.trace(conf))/float(np.sum(conf))

                print '{} runs in {} sec'.format(run+1, time.time() - start_time)
            predictions_fold /= nb_runs

            if early_weights:
                np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/early_val_preds_fold{}.npy".format(i+1), predictions_fold)
            else:
                np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/late_val_preds_fold{}.npy".format(i+1), predictions_fold)
            return
        #     preds += list(predictions_fold)
        # preds = np.array(preds)
        

def localized_run(nb_epoch, train=False, pred_val=False, early_weights=True):
    X, y, X_folds, y_folds, filename_folds = utils.load_data_cropped(fold_file='/home/mzhao/Desktop/kaggle/ncfm/runs/run1/folds.json',
                                                        fish_types=fish_types, size=vgg_size,
                                                        # saved=True, savefileX='X_cropped2_borderless_270x480.npy', savefileY='y_cropped2_borderless_270x480.npy')
                                                        saved=True, savefileX='X_all_cropped2_borderless_270x480.npy', savefileY='y_all_cropped2_borderless_270x480.npy')
    # return

    if train:
        for i in range(k):
            # if i == 0 or i == 1:
            #     continue
            start_time = time.time()
            X_trn = []
            X_val = []
            y_trn = []
            y_val = []
            for j in range(k):
                if j == i:
                    X_val += list(X_folds[j])
                    y_val += list(y_folds[j])
                else:
                    X_trn += list(X_folds[j])
                    y_trn += list(y_folds[j])
            X_trn = np.array(X_trn)
            X_val = np.array(X_val)
            y_trn = np.array(y_trn)
            y_val = np.array(y_val)

            # print X_trn.shape, X_val.shape, y_trn.shape, y_val.shape

            model = models.vgg16()
            trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, size=vgg_size, batch_size=16)

            print "[START] training localized fold {}/{}".format(i+1,k)

            checkpointer = ModelCheckpoint(filepath="/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold{}".format(i+1)+
                                                    "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                            verbose=1, save_weights_only=True)
            model.fit_generator(trn_gen, samples_per_epoch=X_trn.shape[0], nb_epoch=nb_epoch, verbose=2,
                    validation_data=val_gen, nb_val_samples=X_val.shape[0], callbacks=[checkpointer])
                    # validation_data=val_gen, nb_val_samples=X_val.shape[0])

            print "[FINISHED] training localized fold {}/{} in {} sec".format(i+1,k,time.time()-start_time)
            return
        # return

    weights = [["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold1/weights.06-0.41.hdf5"],
               ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold2/weights.06-0.36.hdf5"],
               ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold3/weights.09-0.33.hdf5"]]

    if pred_val:
        for i in range(k):
            start_time = time.time()
            X_trn = []
            X_val = []
            y_trn = []
            y_val = []
            for j in range(k):
                if j == i:
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
                
            if early_weights:
                model.load_weights(weights[i][0])
            else:
                model.load_weights(weights[i][0])

            predictions_fold = np.zeros((X_val.shape[0], nb_classes))
            for run in range(nb_runs):
                print("Starting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
                trn_gen, val_gen = models.get_train_val_gens(X_trn=X_trn, X_val=X_val, y_trn=y_trn, y_val=y_val, size=vgg_size, batch_size=16)
                predictions_run = model.predict_generator(val_gen, val_samples=X_val.shape[0])
                predictions_fold += predictions_run

                print np.sum(y_val, axis=0)
                print np.sum(predictions_run, axis=0)

                actual_pred = np.zeros(y_val.shape[0])
                actual_label = np.zeros(predictions_fold.shape[0])
                for ii in range(y_val.shape[0]):
                    actual_pred[ii] = np.argmax(y_val[ii])
                    actual_label[ii] = np.argmax(predictions_run[ii])
                conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
                print conf
                print float(np.trace(conf))/float(np.sum(conf))

                print '{} runs in {} sec'.format(run+1, time.time() - start_time)
            predictions_fold /= nb_runs

            if early_weights:
                np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/early_val_preds_fold{}.npy".format(i+1), predictions_fold)
            else:
                np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/late_val_preds_fold{}.npy".format(i+1), predictions_fold)
            return
            # preds += list(predictions_fold)
        # preds = np.array(preds)



def stacked_run():
    X, y, X_folds, y_folds, filename_folds = utils.load_data(fish_types=fish_types, size=vgg_size,
                                        saved=True, savefileX='X.npy', savefileY='y.npy', k=3)

    val_pred_files = ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/early_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/early_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/late_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/late_val_preds_fold{}.npy"]

    # legit = []
    # for i in range(k):
    #     legit += list(y_folds[i])
    # legit = np.array(legit)
    legit = y_folds[0]
    preds1 = np.load(val_pred_files[0].format(1))
    preds2 = np.load(val_pred_files[1].format(1))
    # preds3 = np.load(val_pred_files[2].format(1))
    # preds4 = np.load(val_pred_files[3].format(1))
    # preds += np.load(val_pred_files[1])
    # preds /= 2

    print np.sum(legit, axis=0)
    print np.sum(preds1, axis=0)
    print np.sum(preds2, axis=0)

    actual_pred = np.zeros(legit.shape[0])
    actual_label = np.zeros(legit.shape[0])
    for i in range(legit.shape[0]):
        actual_pred[i] = np.argmax(preds1[i])
        # actual_label[i] = np.argmax(preds2[i])
        actual_label[i] = np.argmax(legit[i])
    conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))

    count1 = 0
    count2 = 0
    for i in range(len(legit)):
        if np.argmax(legit[i]) == np.argmax(preds1[i]) or np.argmax(legit[i]) == np.argmax(preds2[i]):
            count1 += 1
        # if np.argmax(legit[i]) == np.argmax(preds3[i]) or np.argmax(legit[i]) == np.argmax(preds4[i]):
        #     count2 += 1
    print count1, count2, len(legit)

    # print actual_pred
    # print actual_label

    exit(1)

    X_stack = np.zeros((X.shape[0],0))
    for vpf in val_pred_files:
        X_stack = np.hstack((X_stack, np.load(vpf)))

    print X_stack.shape



if __name__ == '__main__':
    # e2e_run(nb_epoch_e2e, train=False, pred_val=True, early_weights=True)
    # localized_run(nb_epoch_localized, train=False, pred_val=True, early_weights=True)
    stacked_run()





