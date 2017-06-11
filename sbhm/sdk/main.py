import numpy as np
import pandas as pd
from feature_engineering.preprocess import import_clean
from feature_engineering.generate_features import \
    encode_categorical_features, \
    generate_time_features,\
    generate_relative_square_footage,\
    generate_room_information
from utils.kfold import KFold
from models.naive_xgb import NaiveXGB
import time
import sys
import math

# maybe I will create support for reading from json config files later

LABEL_NAME = 'price_doc'
NUM_FOLDS = 5

def generate_data():
    train, test, macro = import_clean()
    encode_categorical_features(train, test)
    generate_time_features(train, test)
    generate_relative_square_footage(train, test)
    generate_room_information(train, test)
    return train, test, macro

def initialize_L0_models():

    models = []
    models.append(NaiveXGB())

    return models

def train_L0(models, train):
    X_trn = train.drop(LABEL_NAME, 1)
    y_trn = train[LABEL_NAME]
    
    kf = KFold(X_trn, y_trn, num_folds=NUM_FOLDS)
    L0_df = pd.DataFrame()
    for model in models:
        print '\tTraining {}...'.format(model.name)
        L0_df[model.name] = kf.run_kfolds_on_model(model)
    return L0_df

def test_L0(models, test):
    kf = KFold(X_trn, y_trn, num_folds=NUM_FOLDS)
    L0_df = pd.DataFrame()
    for model in models:
        print '\tTraining {}...'.format(model.name)
        L0_df[model.name] = kf.run_kfolds_on_model(model)
    return L0_df

def validate(preds, labels, loss='RMSLE'):
    assert len(preds) == len(labels)

    total = 0
    for pred, label in zip(preds, labels):
        if loss == 'RMSLE':
            total += (math.log(pred+1) - math.log(label+1))**2
        else:
            print '[ERROR] Unknown loss: {}'.format(loss)
            sys.exit(1)
    print 'Validation loss is {:.5f}'.format(total / len(preds))

def run():
    t = time.time()

    # feature engineering
    print 'Importing and generating features...'
    train, test, macro = generate_data()
    print 'Finished in {:.2f} seconds.'.format(time.time()-t)
    t = time.time()
    
    # get models ready
    print 'Initializing models...'
    models = initialize_L0_models()
    print 'Finished in {:.2f} seconds.'.format(time.time()-t)
    t = time.time()

    # train
    print 'Training L0 models with {} folds...'.format(NUM_FOLDS)
    L0_df = train_L0(models, train)
    print 'Finished L0 training in {:.2f} seconds.'.format(time.time()-t)
    t = time.time()
    
    # validation
    y_val = L0_df['NaiveXGB']
    validate(y_val, train[LABEL_NAME])
    
    # test



# boilerplate code
if __name__ == '__main__':
    run()
