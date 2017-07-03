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

def generate_properties_per_loc(train, test):
    train['latlon_sum'] = train.groupby(['key']).transform(len)
    test['latlon_sum'] = test.groupby(['key']).transform(len)



def adjust_for_inflation(train, test):

    rate_2016_q2 = 1
    rate_2016_q1 = rate_2016_q2 / .99903
    rate_2015_q4 = rate_2016_q1 / .9831
    rate_2015_q3 = rate_2015_q4 / .9834
    rate_2015_q2 = rate_2015_q3 / .9815
    rate_2015_q1 = rate_2015_q2 / .9932
    rate_2014_q4 = rate_2015_q1 / 1.0112
    rate_2014_q3 = rate_2014_q4 / 1.0169
    rate_2014_q2 = rate_2014_q3 / 1.0086
    rate_2014_q1 = rate_2014_q2 / 1.0126
    rate_2013_q4 = rate_2014_q1 / 0.9902
    rate_2013_q3 = rate_2013_q4 / 1.0041
    rate_2013_q2 = rate_2013_q3 / 1.0044
    rate_2013_q1 = rate_2013_q2 / 1.0104
    rate_2012_q4 = rate_2013_q1 / 0.9832
    rate_2012_q3 = rate_2012_q4 / 1.0277
    rate_2012_q2 = rate_2012_q3 / 1.0279
    rate_2012_q1 = rate_2012_q2 / 1.0279
    rate_2011_q4 = rate_2012_q1 / 1.076
    rate_2011_q3 = rate_2011_q4 / 1.0236
    rate_2011_q2 = rate_2011_q3 / 1
    rate_2011_q1 = rate_2011_q2 / 1.011

    # test data
    test['average_q_price'] = 1

    test_2016_q2_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month <= 7].index
    test.loc[test_2016_q2_index, 'average_q_price'] = rate_2016_q2
    # test.loc[test_2016_q2_index, 'year_q'] = '2016_q2'

    test_2016_q1_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 1].loc[test['timestamp'].dt.month < 4].index
    test.loc[test_2016_q1_index, 'average_q_price'] = rate_2016_q1
    # test.loc[test_2016_q2_index, 'year_q'] = '2016_q1'

    test_2015_q4_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 10].loc[test['timestamp'].dt.month < 12].index
    test.loc[test_2015_q4_index, 'average_q_price'] = rate_2015_q4
    # test.loc[test_2015_q4_index, 'year_q'] = '2015_q4'

    test_2015_q3_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 7].loc[test['timestamp'].dt.month < 10].index
    test.loc[test_2015_q3_index, 'average_q_price'] = rate_2015_q3
    # test.loc[test_2015_q3_index, 'year_q'] = '2015_q3'

    # test_2015_q2_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index
    # test.loc[test_2015_q2_index, 'average_q_price'] = rate_2015_q2

    # test_2015_q1_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index
    # test.loc[test_2015_q1_index, 'average_q_price'] = rate_2015_q1


    # train 2015
    train['average_q_price'] = 1

    train_2015_q4_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    # train.loc[train_2015_q4_index, 'price_doc'] = train.loc[train_2015_q4_index, 'price_doc'] * rate_2015_q4
    train.loc[train_2015_q4_index, 'average_q_price'] = rate_2015_q4

    train_2015_q3_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    #train.loc[train_2015_q3_index, 'price_doc'] = train.loc[train_2015_q3_index, 'price_doc'] * rate_2015_q3
    train.loc[train_2015_q3_index, 'average_q_price'] = rate_2015_q3

    train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    #train.loc[train_2015_q2_index, 'price_doc'] = train.loc[train_2015_q2_index, 'price_doc'] * rate_2015_q2
    train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

    train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    #train.loc[train_2015_q1_index, 'price_doc'] = train.loc[train_2015_q1_index, 'price_doc'] * rate_2015_q1
    train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


    # train 2014
    train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    #train.loc[train_2014_q4_index, 'price_doc'] = train.loc[train_2014_q4_index, 'price_doc'] * rate_2014_q4
    train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

    train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    #train.loc[train_2014_q3_index, 'price_doc'] = train.loc[train_2014_q3_index, 'price_doc'] * rate_2014_q3
    train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

    train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    #train.loc[train_2014_q2_index, 'price_doc'] = train.loc[train_2014_q2_index, 'price_doc'] * rate_2014_q2
    train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

    train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    #train.loc[train_2014_q1_index, 'price_doc'] = train.loc[train_2014_q1_index, 'price_doc'] * rate_2014_q1
    train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1


    # train 2013
    train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    # train.loc[train_2013_q4_index, 'price_doc'] = train.loc[train_2013_q4_index, 'price_doc'] * rate_2013_q4
    train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

    train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    # train.loc[train_2013_q3_index, 'price_doc'] = train.loc[train_2013_q3_index, 'price_doc'] * rate_2013_q3
    train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

    train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    # train.loc[train_2013_q2_index, 'price_doc'] = train.loc[train_2013_q2_index, 'price_doc'] * rate_2013_q2
    train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

    train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    # train.loc[train_2013_q1_index, 'price_doc'] = train.loc[train_2013_q1_index, 'price_doc'] * rate_2013_q1
    train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1


    # train 2012
    train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    # train.loc[train_2012_q4_index, 'price_doc'] = train.loc[train_2012_q4_index, 'price_doc'] * rate_2012_q4
    train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

    train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    # train.loc[train_2012_q3_index, 'price_doc'] = train.loc[train_2012_q3_index, 'price_doc'] * rate_2012_q3
    train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

    train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    # train.loc[train_2012_q2_index, 'price_doc'] = train.loc[train_2012_q2_index, 'price_doc'] * rate_2012_q2
    train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

    train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    # train.loc[train_2012_q1_index, 'price_doc'] = train.loc[train_2012_q1_index, 'price_doc'] * rate_2012_q1
    train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


    # train 2011
    train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
    # train.loc[train_2011_q4_index, 'price_doc'] = train.loc[train_2011_q4_index, 'price_doc'] * rate_2011_q4
    train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

    train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
    # train.loc[train_2011_q3_index, 'price_doc'] = train.loc[train_2011_q3_index, 'price_doc'] * rate_2011_q3
    train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

    train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
    # train.loc[train_2011_q2_index, 'price_doc'] = train.loc[train_2011_q2_index, 'price_doc'] * rate_2011_q2
    train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

    train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
    # train.loc[train_2011_q1_index, 'price_doc'] = train.loc[train_2011_q1_index, 'price_doc'] * rate_2011_q1
    train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

    #train['price_doc'] = train['price_doc'] * train['average_q_price']
    # train.drop('average_q_price', axis=1, inplace=True)