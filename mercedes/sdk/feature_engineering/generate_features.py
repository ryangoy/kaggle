import numpy as np
import pandas as pd
from sklearn import preprocessing

# TO DO per dataset:
# - write as many feature generation methods as desired
# - call the methods in generate_features

def generate_features(train, test):
    # call desired feature generation methods here
    encode_categorical_features(train, test)

def encode_categorical_features(train, test):
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[c].values)) 
            train[c] = lbl.transform(list(train[c].values))
            train.drop(c, axis=1, inplace=True)
            
    for c in test.columns:
        if test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(test[c].values)) 
            test[c] = lbl.transform(list(test[c].values))
            test.drop(c, axis=1, inplace=True)  

# generic time feature generation from data objects (from SBHM kaggle competition)
def generate_time_features(train, test):   
    # Add month-year
    month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)

    month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek

    test['month'] = test.timestamp.dt.month
    test['dow'] = test.timestamp.dt.dayofweek

    train.drop('timestamp', axis=1, inplace=True)
    test.drop('timestamp', axis=1, inplace=True)
