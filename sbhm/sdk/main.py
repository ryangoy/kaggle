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
-Indexing is assumed to be conserved. If a method shuffles the data,
it returns predictions in the original order.
-Many things are still hard coded at this point. There are some features
that are in here specific to the Sberbank Kaggle competition. It shouldn't
be too hard to adjust, though
-At test time, predictions are generated by running the test data through
each trained fold and averaged over the number of folds at the end.
-I might add json config file support so defining everything becomes more
simple and readable.
"""

import numpy as np
import pandas as pd
from feature_engineering.preprocess import import_clean
from feature_engineering.generate_features import \
    encode_categorical_features, \
    generate_time_features,\
    generate_relative_square_footage,\
    generate_room_information, \
    adjust_for_inflation
from utils.kfold import KFold
from models.naive_xgb import NaiveXGB
from models.light_gbm import LightGBM
from models.neural_net import NeuralNet
from models.elastic_net import ElasticNet
import time
import sys
import math

LABEL_NAME = 'price_doc'
NUM_FOLDS = 5
SUBMISSION_PATH = 'sub.csv'

def generate_data():
    train, test = import_clean()
    adjust_for_inflation(train, test)
    encode_categorical_features(train, test)
    generate_time_features(train, test)
    generate_relative_square_footage(train, test)
    generate_room_information(train, test)
    
    return train, test

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
    L0_models.append(NeuralNet(features = ['full_sq', 'floor', 'build_year', 'life_sq',
        'micex_cbi_tr', 'max_floor', 'eurrub', 'green_zone_km']))
    L0_models.append(LightGBM())
    bruno_params = {
        'min_child_weight': 3,
        'grow_policy': 'lossguide',
        'max_leaves': 30,
        'max_bins': 512,
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    L0_models.append(NaiveXGB(name='BrunoModel', xgb_params=bruno_params, log_data=True))
    L0_models.append(ElasticNet(features = 
        ['full_sq', 'life_sq', 'floor', 'build_year', 'num_room', 'kitch_sq', 'full_all']))

    L0_models.append(LightGBM(log_data=True))

    xgb_params = {
        'eta': 0.1,
        'max_depth': 7,
        'objective': 'reg:linear',
        'silent':1
    }
    L0_models.append(NaiveXGB(xgb_params=xgb_params, log_data=True))
    levels.append(L0_models)
    

    #############
    # L1 MODELS #
    #############
    L1_models = []
    L1_models.append(NaiveXGB(xgb_params=bruno_params,name="L1", log_data=True))
    levels.append(L1_models)

    return levels


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
    return val_df, test_df

def validate(preds, labels, loss='RMSLE'):
    assert len(preds) == len(labels)

    total = 0
    for pred, label in zip(preds, labels):
        if loss == 'RMSLE':
            total += (math.log(pred+1) - math.log(label+1))**2
        else:
            print '[ERROR] Unknown loss: {}'.format(loss)
            sys.exit(1)
    print 'Validation loss is {:.5f}'.format((total / len(preds))**0.5)

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
    
    # validation
    y_val = train.iloc[:,-1] # assuming the last feature is now the last prediction
    validate(y_val, train[LABEL_NAME])

    # write submission file
    submission_df = pd.DataFrame()
    submission_df['id'] = test['id']
    submission_df[LABEL_NAME] = test.iloc[:,-1]
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print 'Total runtime: {:.2f} seconds'.format(time.time()-start_time)

# boilerplate code
if __name__ == '__main__':
    run()
