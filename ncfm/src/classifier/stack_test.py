import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
import time
import xgboost as xgb

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


best = np.load("pred/pred_vgg16_cropped22.npy")
p2 = np.load("pred/pred_vgg16_all_10epochs_relabeled.npy")
best1 = np.array(p2)
for i in range(len(best)):
    best[i] *= (1 - p2[i][4])
best = np.insert(best, 4, p2[:,4], axis=1)
best2 = np.array(best)
best += p2
best /= 2


def e2e_run(nb_epoch, train=False, pred_val=False, pred_test=False, early_weights=True):
    print '[STARTING END TO END RUN]'
    X, y, X_folds, y_folds, filename_folds = utils.load_data(fold_file='/home/mzhao/Desktop/kaggle/ncfm/runs/run1/folds.json',
                                        fish_types=fish_types, size=vgg_size,
                                        saved=False, savefileX='X2.npy', savefileY='y2.npy', k=k)
    # return

    # for i in range(len(X_folds[0])):
    #     print filename_folds[0][i], np.argmax(y_folds[0][i])
    #     print X_folds[0][i][0][0][0], X_folds[0][i].transpose(1,2,0)[0][0][0]
    #     plt.imshow(X_folds[0][i].transpose(1,2,0)[:,:,::-1])
    #     plt.show()

    if train:
        print "[TRAINING END TO END MODEL]"
        for i in range(k):
            # if i == 0:
                # continue
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
            # return
        # return

    weights = [["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold1/weights.09-0.24.hdf5",
                "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold1/weights.13-0.21.hdf5"],
               ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold2/weights.09-0.34.hdf5",
                "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold2/weights.13-0.33.hdf5"],
               ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold3/weights.06-0.36.hdf5",
                "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/fold3/weights.14-0.26.hdf5"]]

    if pred_val:
        print "[PREDICTING ON VALIDATION - END TO END]"
        # preds = []
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

            model = models.vgg16()
                
            if early_weights:
                model.load_weights(weights[i][0])
            else:
                model.load_weights(weights[i][-1])

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
            break
        #     preds += list(predictions_fold)
        # preds = np.array(preds)
    if pred_test:
        print "[PREDICTING ON TEST - END TO END"
        X_test = []
        for file in sorted(os.listdir("test/unknown")):
            path = "test/unknown/{}".format(file)
            img = cv2.imread(path)
            img = cv2.resize(img, (vgg_size[1], vgg_size[0]), cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            X_test += [img]
        X_test = np.array(X_test)

        preds = np.zeros((nb_test_samples, nb_classes))
        for i in range(k):
            start_time = time.time()
            # if i == 0 or i == 1:
            #     continue

            model = models.vgg16()
                
            if early_weights:
                model.load_weights(weights[i][0])
            else:
                model.load_weights(weights[i][-1])

            predictions_fold = np.zeros(preds.shape)
            for run in range(nb_runs):
                print("Starting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
                test_gen = models.get_test_gens(X_test=X_test, size=vgg_size, batch_size=16)
                predictions_run = model.predict_generator(test_gen, val_samples=nb_test_samples)
                predictions_fold += predictions_run
                print np.sum(predictions_run, axis=0)
                print '{} runs in {} sec'.format(run+1, time.time() - start_time)

                actual_pred = np.zeros(predictions_run.shape[0])
                actual_label = np.zeros(predictions_run.shape[0])
                for ii in range(best.shape[0]):
                    actual_pred[ii] = np.argmax(best[ii])
                    actual_label[ii] = np.argmax(predictions_run[ii])
                conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
                print conf
                print float(np.trace(conf))/float(np.sum(conf))

            predictions_fold /= nb_runs
            preds += predictions_fold

            if early_weights:
                np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/early_test_preds_fold{}.npy".format(i+1), predictions_fold)
            else:
                np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/late_test_preds_fold{}.npy".format(i+1), predictions_fold)
        

        preds /= k

        if early_weights:
            np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/early_test_preds.npy", preds)
        else:
            np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/late_test_preds.npy", preds)
        

def localized_run(nb_epoch, train=False, pred_val=False, pred_test=False, early_weights=True):
    print '[STARTING LOCALIZED RUN]'
    X, y, X_folds, y_folds, filename_folds = utils.load_data_cropped(fold_file='/home/mzhao/Desktop/kaggle/ncfm/runs/run1/folds.json',
                                                        fish_types=fish_types, size=vgg_size,
                                                        # saved=False, savefileX='X_cropped2_borderless_270x480.npy', savefileY='y_cropped2_borderless_270x480.npy')
                                                        saved=False, savefileX='X_all_cropped2_borderless_270x480.npy', savefileY='y_all_cropped2_borderless_270x480.npy', k=k)
    # return

    # for i in range(len(X_folds[0])):
    #     print filename_folds[0][i], np.argmax(y_folds[0][i])
    #     print X_folds[0][i][0][0][0], X_folds[0][i].transpose(1,2,0)[0][0][0]
    #     plt.imshow(X_folds[0][i].transpose(1,2,0)[:,:,::-1])
    #     plt.show()

    if train:
        print "[TRAINING LOCALIZED MODEL]"
        for i in range(k):
            # if i == 0:
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
            # return
        # return

    weights = [["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold1/weights.08-0.38.hdf5",
                "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold1/weights.13-0.22.hdf5"],
               ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold2/weights.05-0.42.hdf5"],
               ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold3/weights.06-0.46.hdf5",
                "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/fold3/weights.12-0.43.hdf5"]]

    if pred_val:
        print "[PREDICTING ON VALIDATION - LOCALIZED]"
        for i in range(k):
            if i == 0:
                continue
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
                model.load_weights(weights[i][-1])

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
            # return
            # preds += list(predictions_fold)
        # preds = np.array(preds)

    if pred_test:
        print "[PREDICTING ON TEST - END TO END"
        X_test = []
        index = 0
        for file in sorted(os.listdir("test_cropped2/unknown")):
            path = "test_cropped2/unknown/{}".format(file)
            img = cv2.imread(path)
            if img.shape[0] > img.shape[1]:
                img = img.transpose((1, 0, 2))
            img = cv2.resize(img, (vgg_size[1], vgg_size[0]), cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            X_test += [img]
        X_test = np.array(X_test)

        preds = np.zeros((nb_test_samples, nb_classes))
        for i in range(k):
            start_time = time.time()
            # if i == 0 or i == 1:
            #     continue

            model = models.vgg16()
                
            if early_weights:
                model.load_weights(weights[i][0])
            else:
                model.load_weights(weights[i][-1])

            predictions_fold = np.zeros(preds.shape)
            for run in range(nb_runs):
                print("Starting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
                test_gen = models.get_test_gens(X_test=X_test, size=vgg_size, batch_size=16)
                predictions_run = model.predict_generator(test_gen, val_samples=nb_test_samples)
                predictions_fold += predictions_run
                print np.sum(predictions_run, axis=0)
                print '{} runs in {} sec'.format(run+1, time.time() - start_time)

                actual_pred = np.zeros(predictions_run.shape[0])
                actual_label = np.zeros(predictions_run.shape[0])
                for ii in range(best.shape[0]):
                    actual_pred[ii] = np.argmax(best[ii])
                    actual_label[ii] = np.argmax(predictions_run[ii])
                conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
                print conf
                print float(np.trace(conf))/float(np.sum(conf))

            predictions_fold /= nb_runs
            preds += predictions_fold

            if early_weights:
                np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/early_test_preds_fold{}.npy".format(i+1), predictions_fold)
            else:
                np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/late_test_preds_fold{}.npy".format(i+1), predictions_fold)
        
        preds /= k

        if early_weights:
            np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/early_test_preds.npy", preds)
        else:
            np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/late_test_preds.npy", preds)
        


def stacked_run(pred_test=False):
    X, y, X_folds, y_folds, filename_folds = utils.load_data(fold_file='/home/mzhao/Desktop/kaggle/ncfm/runs/run1/folds.json',
                                        fish_types=fish_types, size=vgg_size,
                                        saved=True, savefileX='X2.npy', savefileY='y2.npy', k=k)

    val_pred_files = ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/early_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/early_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/late_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/late_val_preds_fold{}.npy"]

    test_pred_files = ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/early_test_preds.npy",
                       "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/early_test_preds.npy"]

    predictions_test1 = np.load(test_pred_files[0])
    predictions_test2 = np.load(test_pred_files[1])
    pred_sum = (predictions_test1 + predictions_test2) / 2
    print np.sum(predictions_test1, axis=0)
    print np.sum(predictions_test2, axis=0)
    print np.sum(pred_sum, axis=0)

    actual_pred = np.zeros(best.shape[0])
    actual_label = np.zeros(best.shape[0])
    for i in range(best.shape[0]):
        actual_pred[i] = np.argmax(predictions_test1[i])
        actual_label[i] = np.argmax(best1[i])
    conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))

    actual_pred = np.zeros(best.shape[0])
    actual_label = np.zeros(best.shape[0])
    for i in range(best.shape[0]):
        actual_pred[i] = np.argmax(predictions_test2[i])
        actual_label[i] = np.argmax(best2[i])
    conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))

    actual_pred = np.zeros(best.shape[0])
    actual_label = np.zeros(best.shape[0])
    for i in range(best.shape[0]):
        actual_pred[i] = np.argmax(pred_sum[i])
        actual_label[i] = np.argmax(best[i])
    conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))


    preds = np.clip(pred_sum, 0.02, .98, out=None)
    print np.sum(preds, axis=0)

    filenames = sorted(os.listdir("test_cropped2/unknown"))
    # print filenames[:10]
    with open('/home/mzhao/Desktop/kaggle/ncfm/runs/run1/submission.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % (p/np.sum(preds[i, :])) for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")

    exit(1)


    if pred_test:
        predictions_test = np.zeros((nb_test_samples, nb_classes))

    for i in range(k):
        X_trn = []
        y_trn = []
        X_val = []
        y_val = []
        for j in range(k):
            if j == i:
                X_val += list(np.hstack((np.load(val_pred_files[0].format(j+1)), np.load(val_pred_files[1].format(j+1)))))
                y_val += list(y_folds[j])
            else:
                X_trn += list(np.hstack((np.load(val_pred_files[0].format(j+1)), np.load(val_pred_files[1].format(j+1)))))
                y_trn += list(y_folds[j])
        X_trn = np.array(X_trn)
        X_val = np.array(X_val)
        y_trn = np.array(y_trn)
        y_val = np.array(y_val)
        X_test = np.hstack((np.load(test_pred_files[0]), np.load(test_pred_files[1])))

        print X_trn.shape, X_val.shape, y_trn.shape, y_val.shape
        print np.sum(y_trn, axis=0)
        print np.sum(y_val, axis=0)

        use_rf = False
        use_xgb = True

        # random forest stack
        if use_rf:
            rf = RandomForestClassifier(n_estimators=1000, max_depth=None, 
                                        min_samples_split=2, min_samples_leaf=1, 
                                        max_leaf_nodes=None, verbose=1)
            rf.fit(X_trn, y_train)
            pred = rf.predict(X_val)

            if pred_test:
                predictions_fold = rf.predict(X_test)
                predictions_test += predictions_fold

        # xgboost stack
        if use_xgb:
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
            xg_trn = xgb.DMatrix(X_trn, label=np.argmax(y_trn, axis=1))
            xg_val = xgb.DMatrix(X_val, label=np.argmax(y_val, axis=1))

            watchlist = [ (xg_trn,'train'), (xg_val, 'val') ]
            model = xgb.train(plst, xg_trn, num_rounds, watchlist, early_stopping_rounds=20)

            pred = model.predict(xg_val)

            if pred_test:
                predictions_fold = model.predict(xgb.DMatrix(X_test))
                predictions_test += predictions_fold

                actual_pred = np.zeros(predictions_test.shape[0])
                actual_label = np.zeros(predictions_test.shape[0])
                for i in range(predictions_test.shape[0]):
                    actual_pred[i] = np.argmax(predictions_fold[i])
                    actual_label[i] = np.argmax(best[i])
                conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
                print conf
                print float(np.trace(conf))/float(np.sum(conf))


        actual_pred = np.zeros(y_val.shape[0])
        actual_label = np.zeros(y_val.shape[0])
        for i in range(y_val.shape[0]):
            actual_pred[i] = np.argmax(pred[i])
            actual_label[i] = np.argmax(y_val[i])
        conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
        print conf
        print float(np.trace(conf))/float(np.sum(conf))

        cross_entropy = 0
        for i in range(y_val.shape[0]):
            for j in range(y_val.shape[1]):
                cross_entropy -= y_val[i][j] * np.log(max(1e-2, pred[i][j]))
        cross_entropy /= y_val.shape[0]
        print 'categorical cross entropy: ', cross_entropy

    if pred_test:
        predictions_test /= k
        print np.sum(predictions_test, axis=0)
        print 'saving test predictions'
        np.save("/home/mzhao/Desktop/kaggle/ncfm/runs/run1/pred_test.npy", predictions_test)

        predictions_test1 = np.load(test_pred_files[0])
        predictions_test2 = np.load(test_pred_files[1])
        print np.sum(predictions_test1, axis=0)
        print np.sum(predictions_test2, axis=0)

        actual_pred = np.zeros(predictions_test.shape[0])
        actual_label = np.zeros(predictions_test.shape[0])
        for i in range(predictions_test.shape[0]):
            actual_pred[i] = np.argmax(predictions_test1[i])
            actual_label[i] = np.argmax(predictions_test2[i])
        conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
        print conf
        print float(np.trace(conf))/float(np.sum(conf))

        preds = np.clip(predictions_test, 0.02, .98, out=None)
        print np.sum(preds, axis=0)

        actual_pred = np.zeros(predictions_test.shape[0])
        actual_label = np.zeros(predictions_test.shape[0])
        for i in range(predictions_test.shape[0]):
            actual_pred[i] = np.argmax(predictions_test1[i])
            actual_label[i] = np.argmax(best[i])
        conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
        print conf
        print float(np.trace(conf))/float(np.sum(conf))

        actual_pred = np.zeros(predictions_test.shape[0])
        actual_label = np.zeros(predictions_test.shape[0])
        for i in range(predictions_test.shape[0]):
            actual_pred[i] = np.argmax(predictions_test2[i])
            actual_label[i] = np.argmax(best[i])
        conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
        print conf
        print float(np.trace(conf))/float(np.sum(conf))

        actual_pred = np.zeros(predictions_test.shape[0])
        actual_label = np.zeros(predictions_test.shape[0])
        for i in range(predictions_test.shape[0]):
            actual_pred[i] = np.argmax(preds[i])
            actual_label[i] = np.argmax(best[i])
        conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred)
        print conf
        print float(np.trace(conf))/float(np.sum(conf))

        vision = []

        filenames = sorted(os.listdir("test_cropped2/unknown"))
        for i in range(30):
            print filenames[i], np.argmax(preds[i]), np.argmax(best[i])

        filenames = sorted(os.listdir("test_cropped2/unknown"))
        # print filenames[:10]
        with open('/home/mzhao/Desktop/kaggle/ncfm/runs/run1/submission.csv', 'w') as f:
            print("Writing Predictions to CSV...")
            f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
            for i, image_name in enumerate(filenames):
                pred = ['%.6f' % (p/np.sum(preds[i, :])) for p in preds[i, :]]
                f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
            print("Done.")

def stacked_stats():
    X, y, X_folds, y_folds, filename_folds = utils.load_data(fold_file='/home/mzhao/Desktop/kaggle/ncfm/runs/run1/folds.json',
                                        fish_types=fish_types, size=vgg_size,
                                        saved=True, savefileX='X2.npy', savefileY='y2.npy', k=k)

    val_pred_files = ["/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/early_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/early_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/e2e/late_val_preds_fold{}.npy",
                      "/home/mzhao/Desktop/kaggle/ncfm/runs/run1/localized/late_val_preds_fold{}.npy"]

    y_stack = []
    pred1 = []
    pred2 = []
    for i in range(k):
        y_stack += list(y_folds[i])
        pred1 += list(np.load(val_pred_files[0].format(i+1)))
        pred2 += list(np.load(val_pred_files[1].format(i+1)))
    y_stack = np.array(y_stack)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    print np.sum(y_stack, axis=0)
    print np.sum(pred1, axis=0)
    print np.sum(pred2, axis=0)

    actual_pred1 = np.zeros(y_stack.shape[0])
    actual_pred2 = np.zeros(y_stack.shape[0])
    actual_label = np.zeros(y_stack.shape[0])
    for i in range(y_stack.shape[0]):
        actual_pred1[i] = np.argmax(pred1[i])
        actual_pred2[i] = np.argmax(pred2[i])
        actual_label[i] = np.argmax(y_stack[i])
    print "conf mat for label vs pred1"
    conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred1)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))
    print "conf mat for label vs pred2"
    conf = sklearn.metrics.confusion_matrix(actual_label, actual_pred2)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))
    print "conf mat for pred1 vs pred2"
    conf = sklearn.metrics.confusion_matrix(actual_pred1, actual_pred2)
    print conf
    print float(np.trace(conf))/float(np.sum(conf))

    count1 = 0
    count2 = 0
    for i in range(len(y_stack)):
        if np.argmax(y_stack[i]) == np.argmax(pred1[i]) or np.argmax(y_stack[i]) == np.argmax(pred2[i]):
            count1 += 1
        # if np.argmax(y_stack[i]) == np.argmax(pred3[i]) or np.argmax(y_stack[i]) == np.argmax(pred4[i]):
        #     count2 += 1
    print count1, count2, len(y_stack)

    cross_entropy = 0
    for i in range(y_stack.shape[0]):
        for j in range(y_stack.shape[1]):
            cross_entropy -= y_stack[i][j] * np.log(max(1e-2, pred1[i][j]))
    cross_entropy /= y_stack.shape[0]
    print 'pred1 - categorical cross entropy: ', cross_entropy

    cross_entropy = 0
    for i in range(y_stack.shape[0]):
        for j in range(y_stack.shape[1]):
            cross_entropy -= y_stack[i][j] * np.log(max(1e-2, pred2[i][j]))
    cross_entropy /= y_stack.shape[0]
    print 'pred2 - categorical cross entropy: ', cross_entropy



if __name__ == '__main__':
    # e2e_run(nb_epoch_e2e, train=False, pred_val=False, pred_test=True, early_weights=True)
    # localized_run(nb_epoch_localized, train=False, pred_val=False, pred_test=True, early_weights=True)
    # e2e_run(nb_epoch_e2e, train=False, pred_val=True, pred_test=True, early_weights=False)
    # localized_run(nb_epoch_localized, train=False, pred_val=True, pred_test=True, early_weights=False)
    # stacked_stats()
    stacked_run(pred_test=True)





