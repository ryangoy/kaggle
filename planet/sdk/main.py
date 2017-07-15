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
from feature_engineering.preprocess import import_clean, generate_binary_data_structure
from feature_engineering.generate_features import \
    generate_features
from utils.kfold import KFold
from models.naive_xgb import NaiveXGB
from models.light_gbm import LightGBM
from models.neural_net import NeuralNet
from models.elastic_net import ElasticNet
from models.extra_trees_regressor import ExtraTreesRegressor
from models.lasso_lars import LassoLars
from models.vgg16_model import VGG16
import time
import sys
import math
from sklearn.metrics import r2_score

# CONSTANTS
LABEL_NAME = 'y'
ID_NAME = 'ID'
NUM_FOLDS = 5
SEED = 0
SUBMISSION_PATH = '/home/ryan/cs/kaggle/planet/submissions/sub.csv'
DS_DIR = '/home/ryan/cs/datasets/planet/'
RELOAD = False # re-generate features
TRAIN_IMAGES = join(DS_DIR, 'original/train-jpg')
TARGET_FOLDER = join(DS_DIR, 'binary')
LABELS_PATH = join(DS_DIR, 'processed/train.csv')
TEST_IMAGES = join(DS_DIR, 'original/test-jpg')
IMAGE_EXTENSION = '.jpg'

def initialize_models():
    """
    All layers of models are defined here. The index of the levels
    array corresponds to the level of the stack. 

    Note: the length of the last level must be 1 (i.e. len(levels[-1]) == 1)
    """
    models = [
        VGG16(classes=2)
    ]
    return models

def print_loss(preds, labels, loss='R2'):
    assert len(preds) == len(labels)
    # TO DO: no loss written for planet dataset yet
    total = 0
    if loss == 'R2':
        total = r2_score(labels, preds)
    elif loss == 'RMSLE': # for custom loss functions
        for pred, label in zip(preds, labels):
            total += (math.log(pred+1) - math.log(label+1))**2
        total = (total / len(preds))**0.5
    else:
        print '[ERROR] Unknown loss: {}'.format(loss)
        sys.exit(1)
    print '{} validation loss is {:.5f}'.format(loss, total)

def generate_data():
    train, test = import_clean(DS_DIR)
    generate_features(train, test)
    return train, test


def train_models(models, labels_df):
    """
    @param models: array of models
    @param train: DataFrame with label columns.
    """
    for model in models:
        print '\tMoving images...'
        generate_binary_data_structure(TRAIN_IMAGES, TARGET_FOLDER, labels_df, 'primary',
                                       extension=IMAGE_EXTENSION)
        print '\tTraining {}...'.format(model.name)
        # history is easily graphable, not doing anything with this yet
        history = model.train(TARGET_FOLDER, None) 
        print '\tFinished training {}.'.format(model.name)

def test_models(models, test):
    test_df = pd.DataFrame()
    for model in models:
        print '\tTesting {}...'.format(model.name)
        test_df[model.name] = model.test(test)
        print '\tFinished testing {}.'.format(model.name)
    return test_df

def write_submission(predictions):
    # TO DO
    return

# boilerplate code
def run():
    t = time.time()
    start_time = t

    print 'Importing and generating features....'
    train, test = generate_data()
    print 'Finished in {:.2f} seconds.'.format(time.time()-t)
    t = time.time()
    
    print 'Initializing models...'
    models = initialize_models()
    print 'Finished in {:.2f} seconds.'.format(time.time()-t)
    t = time.time()

    print 'Training models...'
    val_df = train_models(models, train)
    print 'Finished in {:2f} seconds'.format(time.time()-start_time)

    print 'Testing models...'
    preds_df = test_models(models, TEST_IMAGES)
    print 'Finished in {:2f} seconds'.format(time.time()-start_time)

    print 'Writing submission...'
    write_submission(preds_df)
    print 'Total runtime: {:.2f} seconds'.format(time.time()-start_time)

if __name__ == '__main__':
    run()
