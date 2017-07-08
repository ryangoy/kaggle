"""
Stacking and KFolds SDK
@author Ryan Goy

This code base is for non-image based Kaggle competitions. Models can
be defined in the models folder and must be imported in this file. The
stack is defined in initialize_model(). Feature engineering can be defined
in the feature_engineering folder and must also be imported in this file. 
You must call the functions in generate_data(). The rest should be 
generalized minus a few hard-coded things.

A few things to note:
- Indexing is assumed to be conserved. If a method shuffles the data,
it returns predictions in the original order.
- Many things are still hard coded at this point. There are some features
that are in here specific to the Sberbank Kaggle competition. It shouldn't
be too hard to adjust, though
- At test time, predictions are generated by running the test data through
each trained fold and averaged over the number of folds at the end.
- I might add json config file support so defining everything becomes more
simple and readable.

TO DO per dataset:
 - change the constants starting at line 45
 - customize levels and their parameters (note: levels is a 2D array)
 - change print_loss to a loss function of the competition
"""

import numpy as np
import pandas as pd
from os.path import join
from feature_engineering.preprocess import import_clean
from feature_engineering.generate_features import \
    generate_features
from utils.kfold import KFold
from models.naive_xgb import NaiveXGB
from models.light_gbm import LightGBM
from models.neural_net import NeuralNet
from models.elastic_net import ElasticNet
import time
import sys
import math
from sklearn.metrics import r2_score

# CONSTANTS
LABEL_NAME = 'y'
ID_NAME = 'ID'
NUM_FOLDS = 5
SUBMISSION_PATH = 'sub.csv'
DS_DIR = '/home/ryan/cs/datasets/mercedes/'
RELOAD = False # re-generate features
TRAIN_RELOAD_PATH = join(DS_DIR, 'train_reload.csv')
TEST_RELOAD_PATH = join(DS_DIR, 'test_reload.csv')

def initialize_models():
    """
    All layers of models are defined here. The index of the levels
    array corresponds to the level of the stack. 

    Note: the length of the last level must be 1 (i.e. len(levels[-1]) == 1)
    """
    levels = []
    #############
    # L0 MODELS #
    #############
    L0_models = []
    params = {
        'n_trees': 200, 
        'eta': 0.005,
        'max_depth': 6,
        'subsample': 0.85,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
        }
    xgb_params = {
        'n_trees': 500, 
        'eta': 0.005,
        'max_depth': 4,
        'subsample': 0.95,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    # L0_models.append(NaiveXGB(xgb_params=params, log_data=False, 
    #                           custom_eval=r2_score, maximize=True, 
    #                           early_stopping_rounds=20))
    L0_models.append(NeuralNet(features=['X0']))
    levels.append(L0_models)
    

    #############
    # L1 MODELS #
    #############
    L1_models = []

    #levels.append(L1_models)

    return levels

def print_loss(preds, labels, loss='R2'):
    assert len(preds) == len(labels)
    total = 0
    if loss == 'R2':
        total = -1*r2_score(labels, preds)
    elif loss == 'RMSLE': # for custom loss functions
        for pred, label in zip(preds, labels):
            total += (math.log(pred+1) - math.log(label+1))**2
        total = (total / len(preds))**0.5
    else:
        print '[ERROR] Unknown loss: {}'.format(loss)
        sys.exit(1)
    print '{} validation loss is {:.5f}'.format(loss, total)

#################
# Do not modify #
#################
def generate_data():
    if RELOAD:
        train, test = import_clean(DS_DIR)
        generate_features(train, test)
        if TRAIN_RELOAD_PATH is not None and TEST_RELOAD_PATH is not None:
            train.to_csv(TRAIN_RELOAD_PATH)
            test.to_csv(TEST_RELOAD_PATH)
    else:
        train = pd.read_csv(TRAIN_RELOAD_PATH)
        test = pd.read_csv(TEST_RELOAD_PATH)
    return train, test

def train_and_test_level(models, train, test):
    X_trn = train.drop(LABEL_NAME, 1)
    y_trn = train[LABEL_NAME]
    
    kf = KFold(X_trn, y_trn, test, num_folds=NUM_FOLDS)
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for model in models:
        print '\tTraining {}...'.format(model.name)
        val_df[model.name], test_df[model.name] = kf.run_kfolds_on_model(model)
        print '\tFinished training {}.'.format(model.name)
        print_loss(val_df[model.name], y_trn)
    return val_df, test_df

def run():
    t = time.time()
    start_time = t

    # feature engineering
    print 'Importing and generating features...'
    train, test = generate_data()
    print 'Finished in {:.2f} seconds.'.format(time.time()-t)
    t = time.time()
    
    # get models ready
    print 'Initializing models...'
    levels = initialize_models()
    print 'Finished in {:.2f} seconds.'.format(time.time()-t)
    t = time.time()

    # train & test for each model
    # this is a weird approach, but it's simpler since we don't have to keep track of models
    # for test time.
    for i in range(len(levels)):
        print 'Training and testing L{} models with {} folds...'.format(i, NUM_FOLDS)
        out_train_df, out_test_df = train_and_test_level(levels[i], train, test)

        train = pd.concat([train, out_train_df], axis=1)
        test = pd.concat([test, out_test_df], axis=1)

        print 'Finished L{} training and testing in {:.2f} seconds.'.format(i, time.time()-t)
        t = time.time()

    # write submission file
    submission_df = pd.DataFrame()
    submission_df[ID_NAME] = test[ID_NAME]
    submission_df[LABEL_NAME] = test.iloc[:,-1]
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print 'Total runtime: {:.2f} seconds'.format(time.time()-start_time)

# boilerplate code
if __name__ == '__main__':
    run()
