import numpy as np
import pandas as pd
from sklearn import preprocessing

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

def generate_relative_square_footage(train, test):
    # relative floor of the apartment
    train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
    test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)

    # relative size of kitchen
    train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)  
    test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

def generate_room_information(train, test):
    # average room size
    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)