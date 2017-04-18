import datetime
import json
import numpy as np
import pandas as pd
import random
from scipy import sparse
import sklearn
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time
import xgboost as xgb

random.seed(0)
np.random.seed(0)

# runs xgboost for validation and test runs
def runXGB(X_trn, y_trn, X_test, y_test=None, feature_names=None, seed_val=0, num_rounds=2000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.03
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = 'mlogloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(X_trn, label=y_trn)

    if y_test is not None:
        xgtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(X_test)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

# one hots categorical features in 'features_to_use'
def vectorize_categorical_features(trn_df, test_df, features_to_use):
    categorical = ['display_address', 'manager_id', 'building_id', 'street_address']
    for f in categorical:
        if f in features_to_use and trn_df[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(trn_df[f].values) + list(test_df[f].values))
            trn_df[f] = lbl.transform(list(trn_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)

# adds custom features in 'features_to_use'
def feature_engineering(trn_df, test_df, features_to_use):
    # total number of rooms
    if 'rooms' in features_to_use:
        trn_df['rooms'] = trn_df['bedrooms']+trn_df['bathrooms'] 
        test_df['rooms'] = test_df['bedrooms']+test_df['bathrooms'] 
    # if rental unit has a half bathroom
    if 'half_bathrooms' in features_to_use:
        trn_df['half_bathrooms'] = trn_df['bathrooms'].apply(lambda x: 1 if x%1==.5 else 0) 
        test_df['half_bathrooms'] = test_df['bathrooms'].apply(lambda x: 1 if x%1==.5 else 0)
    # price per bedroom
    if 'price_t' in features_to_use:
        trn_df['price_t'] = trn_df['price']/trn_df['bedrooms']
        test_df['price_t'] = test_df['price']/test_df['bedrooms'] 
    # price per bathroom
    if 'price_s' in features_to_use:
        trn_df['price_s'] = trn_df['price']/trn_df['bathrooms']
        test_df['price_s'] = test_df['price']/test_df['bathrooms']
    # price per room
    if 'price_r' in features_to_use:
        trn_df['price_r'] = trn_df['price']/trn_df['rooms']
        test_df['price_r'] = test_df['price']/test_df['rooms'] 
    # number of photos
    if 'num_photos' in features_to_use:
        trn_df['num_photos'] = trn_df['photos'].apply(len)
        test_df['num_photos'] = test_df['photos'].apply(len)
    # number of rental unit features (parking, pets, etc)
    if 'num_features' in features_to_use:
        trn_df['num_features'] = trn_df['features'].apply(len)
        test_df['num_features'] = test_df['features'].apply(len)
    # number of words in description
    if 'num_description_words' in features_to_use:
        trn_df['num_description_words'] = trn_df['description'].apply(lambda x: len(x.split(' ')))
        test_df['num_description_words'] = test_df['description'].apply(lambda x: len(x.split(' ')))
    # percentage of year passed when listing created
    if 'created_year_percent' in features_to_use:
        def year_percent(date_unicode):
            date = datetime.datetime.strptime(date_unicode, '%Y-%m-%d %H:%M:%S')
            day = date.date()
            year_start = datetime.date(date.year, 1, 1)
            year_end = datetime.date(date.year, 12, 31)
            year_portion = time.mktime(day.timetuple()) - time.mktime(year_start.timetuple())
            year_total = time.mktime(year_end.timetuple()) - time.mktime(year_start.timetuple())
            return year_portion/year_total
        trn_df['created_year_percent'] = trn_df['created'].apply(lambda x: year_percent(x))
        test_df['created_year_percent'] = test_df['created'].apply(lambda x: year_percent(x))
    # percentage through train data when listing created
    if 'created_percent' in features_to_use:
        time_start = None
        time_end = None
        for i in range(len(trn_df.index)):
            date_unicode = trn_df['created'].iloc[i]
            date = datetime.datetime.strptime(date_unicode, '%Y-%m-%d %H:%M:%S')
            day = date.date()
            cur = time.mktime(day.timetuple())
            if time_start == None or cur < time_start:
                time_start = cur
            if time_end == None or cur > time_end:
                time_end = cur
        # print 'time start: ', time_start
        # print 'time end: ', time_end
        # print 'total time period: ', time_end - time_start
        def created_percent(date_unicode):
            date = datetime.datetime.strptime(date_unicode, '%Y-%m-%d %H:%M:%S')
            day = date.date()
            time_portion = max(min(time.mktime(day.timetuple()) - time_start, time_end), time_start)
            time_total = time_end - time_start
            return time_portion/time_total
        trn_df['created_percent'] = trn_df['created'].apply(lambda x: created_percent(x))
        test_df['created_percent'] = test_df['created'].apply(lambda x: created_percent(x))
    # hour at which listing was created
    if 'created_hour' in features_to_use:
        def created_hour(date_unicode):
            date = datetime.datetime.strptime(date_unicode, '%Y-%m-%d %H:%M:%S')
            return date.hour
        trn_df['created_hour'] = trn_df['created'].apply(lambda x: created_hour(x))
        test_df['created_hour'] = test_df['created'].apply(lambda x: created_hour(x))

    # add average image size

# my version of cv stats - overfits horridly since there is too much information on manager for sparse managers
def cv_stats(trn_df, test_df, features_to_use):
    if 'manager_level_low' in features_to_use:
        l_trn = [np.nan]*len(trn_df)
        m_trn = [np.nan]*len(trn_df)
        h_trn = [np.nan]*len(trn_df)

        l_test = [np.nan]*len(test_df)
        m_test = [np.nan]*len(test_df)
        h_test = [np.nan]*len(test_df)

        manager_level = {}
        for manager in trn_df['manager_id'].values:
            manager_level[manager] = [0,0,0]
        for i in range(len(trn_df)):
            temp = trn_df.iloc[i]
            if temp['interest_level'] == 'low':
                index = 0
            if temp['interest_level'] == 'medium':
                index = 1
            if temp['interest_level'] == 'high':
                index = 2
            manager_level[temp['manager_id']][index] += 1
        for i in range(len(trn_df)):
            temp = trn_df.iloc[i]
            levels = manager_level[temp['manager_id']]
            if temp['interest_level'] == 'low':
                index = 0
            if temp['interest_level'] == 'medium':
                index = 1
            if temp['interest_level'] == 'high':
                index = 2
            levels[index] -= 1
            if sum(levels) != 0:
                l_trn[i] = levels[0]*1.0/sum(levels)
                m_trn[i] = levels[1]*1.0/sum(levels)
                h_trn[i] = levels[2]*1.0/sum(levels)
        for i in range(len(test_df)):
            temp = test_df.iloc[i]
            # print "aye" if temp['manager_id'] in manager_level.keys() else "rip"
            if temp['manager_id'] not in manager_level.keys() or sum(manager_level[temp['manager_id']]) == 0:
                l_test[i] = np.nan
                m_test[i] = np.nan
                h_test[i] = np.nan
            else:
                levels = manager_level[temp['manager_id']]
                l_test[i] = levels[0]*1.0/sum(levels)
                m_test[i] = levels[1]*1.0/sum(levels)
                h_test[i] = levels[2]*1.0/sum(levels)

        trn_df['manager_level_low'] = l_trn
        trn_df['manager_level_medium'] = m_trn
        trn_df['manager_level_high'] = h_trn

        test_df['manager_level_low'] = l_test
        test_df['manager_level_medium'] = m_test
        test_df['manager_level_high'] = h_test

# original version of cv stats, better as it uses only part of data to validate against each sample
def cv_stats2(trn_df, test_df, features_to_use):
    if 'manager_level_low' in features_to_use:

        splits = 5

        indicies = list(range(trn_df.shape[0]))
        random.shuffle(indicies)
        l_trn = [np.nan]*len(trn_df)
        m_trn = [np.nan]*len(trn_df)
        h_trn = [np.nan]*len(trn_df)

        for i in range(splits):
            manager_level = {}
            for manager in trn_df['manager_id'].values:
                manager_level[manager] = [0,0,0]
            test_index=indicies[int((i*trn_df.shape[0])/splits):int(((i+1)*trn_df.shape[0])/splits)]
            # trn_index=indicies[int((((i+1)%splits)*trn_df.shape[0])/splits):int((((i+1)%splits+1)*trn_df.shape[0])/splits)]
            trn_index = list(set(indicies).difference(test_index))
            for j in trn_index:
                temp = trn_df.iloc[j]
                if temp['interest_level']=='low':
                    index = 0
                if temp['interest_level']=='medium':
                    index = 1
                if temp['interest_level']=='high':
                    index = 2
                manager_level[temp['manager_id']][index] += 1
            for j in test_index:
                temp = trn_df.iloc[j]
                levels = manager_level[temp['manager_id']]
                if sum(levels)!=0:
                    l_trn[j]=levels[0]*1.0/sum(levels)
                    m_trn[j]=levels[1]*1.0/sum(levels)
                    h_trn[j]=levels[2]*1.0/sum(levels)
        trn_df['manager_level_low'] = l_trn
        trn_df['manager_level_medium'] = m_trn
        trn_df['manager_level_high'] = h_trn

        l_test = [np.nan]*len(test_df)
        m_test = [np.nan]*len(test_df)
        h_test = [np.nan]*len(test_df)
        manager_level = {}
        for j in trn_df['manager_id'].values:
            manager_level[j]=[0,0,0]
        for j in range(trn_df.shape[0]):
            temp = trn_df.iloc[j]
            if temp['interest_level'] == 'low':
                index = 0
            if temp['interest_level'] == 'medium':
                index = 1
            if temp['interest_level'] == 'high':
                index = 2
            manager_level[temp['manager_id']][index]+=1
        for j in range(test_df.shape[0]):
            temp = test_df.iloc[j]
            if temp['manager_id'] not in manager_level.keys():
                l_test[j] = np.nan
                m_test[j] = np.nan
                h_test[j] = np.nan
            else:
                levels = manager_level[temp['manager_id']]
                l_test[j] = levels[0]*1.0/sum(levels)
                m_test[j] = levels[1]*1.0/sum(levels)
                h_test[j] = levels[2]*1.0/sum(levels)
        test_df['manager_level_low'] = l_test
        test_df['manager_level_medium'] = m_test
        test_df['manager_level_high'] = h_test


def run_validation(trn_df, val_df, features_to_use):
    start_time = time.time()
    print "[START] validation run"

    vectorize_categorical_features(trn_df, val_df, features_to_use)
    print '[TIME] to vectorize categorical features:', time.time() - start_time

    feature_engineering(trn_df, val_df, features_to_use)
    print '[TIME] to engineer features:', time.time() - start_time

    cv_stats2(trn_df, val_df, features_to_use)
    print '[TIME] to engineer cv stat features:', time.time() - start_time

    trn_df['features'] = trn_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    val_df['features'] = val_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    print(trn_df['features'].head())
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    trn_sparse = tfidf.fit_transform(trn_df['features'])
    val_sparse = tfidf.transform(val_df['features'])

    print '[TIME] to split rental features into separate features:', time.time() - start_time

    X_trn = sparse.hstack([trn_df[features_to_use], trn_sparse]).tocsr()
    X_val = sparse.hstack([val_df[features_to_use], val_sparse]).tocsr()

    target_num_map = {'low':0, 'medium':1, 'high':2}
    y_trn = np.array(trn_df['interest_level'].apply(lambda x: target_num_map[x]))
    y_val = np.array(val_df['interest_level'].apply(lambda x: target_num_map[x]))
    print X_trn.shape, X_val.shape, y_trn.shape, y_val.shape

    print '[TIME] to create train/validation matrices:', time.time() - start_time

    preds, model = runXGB(X_trn, y_trn, X_val, y_val)

    print '[TIME] to run xgboost:', time.time() - start_time

    fold_log_loss = log_loss(y_val, preds)
    fold_accuracy = accuracy_score(y_val, np.argmax(preds, axis=1))

    print 'confusion matrix:\n', confusion_matrix(y_val, np.argmax(preds, axis=1))
    print 'log loss:', fold_log_loss
    print 'accuracy:', fold_accuracy

    print "[FINISH] validation run"

    return fold_log_loss, fold_accuracy


def run_test(trn_df, test_df, features_to_use):
    start_time = time.time()
    print "[START] test run"

    vectorize_categorical_features(trn_df, test_df, features_to_use)
    print '[TIME] to vectorize categorical features:', time.time() - start_time

    feature_engineering(trn_df, test_df, features_to_use)
    print '[TIME] to engineer features:', time.time() - start_time

    cv_stats2(trn_df, test_df, features_to_use)
    print '[TIME] to engineer cv stat features:', time.time() - start_time

    trn_df['features'] = trn_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    test_df['features'] = test_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    print(trn_df['features'].head())
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    trn_sparse = tfidf.fit_transform(trn_df['features'])
    test_sparse = tfidf.transform(test_df['features'])

    print '[TIME] to split rental features into separate features:', time.time() - start_time

    X_trn = sparse.hstack([trn_df[features_to_use], trn_sparse]).tocsr()
    X_test = sparse.hstack([test_df[features_to_use], test_sparse]).tocsr()

    target_num_map = {'low':0, 'medium':1, 'high':2}
    y_trn = np.array(trn_df['interest_level'].apply(lambda x: target_num_map[x]))
    print X_trn.shape, X_test.shape, y_trn.shape

    print '[TIME] to create train/validation matrices:', time.time() - start_time

    preds, model = runXGB(X_trn, y_trn, X_test, num_rounds=1000)

    print '[TIME] to run xgboost:', time.time() - start_time

    out_df = pd.DataFrame(preds)
    out_df.columns = ["low", "medium", "high"]
    # clipping perhaps
    # for i in range(len(out_df)):
    #     out_df.iloc[i]['low'] = max(out_df.iloc[i]['low'], .02)
    #     out_df.iloc[i]['medium'] = max(out_df.iloc[i]['medium'], .02)
    #     out_df.iloc[i]['high'] = max(out_df.iloc[i]['high'], .02)
    # # print out_df
    # for i in range(len(out_df)):
    #     out_df.iloc[i]['low'] /= sum(out_df.iloc[i])
    #     out_df.iloc[i]['medium'] /= sum(out_df.iloc[i])
    #     out_df.iloc[i]['high'] /= sum(out_df.iloc[i])
    # print out_df
    out_df["listing_id"] = test_df.listing_id.values
    out_df.to_csv("xgb_basic_cv_stat.csv", index=False)

    print '[TIME] to created submission:', time.time() - start_time

    print '[FINISH] test run'


if __name__ == '__main__':
    trn_all_df = pd.read_json('data/train.json')
    test_df=pd.read_json('data/test.json')

    features_to_use = [ 
                        # numerical
                        'bathrooms', 
                        'bedrooms', 
                        'latitude', 
                        'longitude', 
                        'price',
                        'listing_id', 
                        # categorical
                        'display_address', 
                        'manager_id', 
                        'building_id', 
                        'street_address',
                        # engineered
                        'rooms', 
                        'half_bathrooms',
                        'price_t', 
                        'price_s', 
                        'price_r', 
                        'num_photos', 
                        'num_features', 
                        'num_description_words',
                        'created_year_percent', 
                        'created_percent', 
                        'created_hour',
                        # cv stats or whatever that means
                        'manager_level_low', 'manager_level_medium', 'manager_level_high',
                      ]

    cv_scores = []
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for trn_index, val_index in kf.split(range(trn_all_df.shape[0])):
        print "VALIDATION FOLD", len(cv_scores) + 1
        trn_df = trn_all_df.iloc[trn_index]
        val_df = trn_all_df.iloc[val_index]
        trn_df.is_copy = False
        val_df.is_copy = False
        cv_scores += [run_validation(trn_df, val_df, features_to_use)]
    for i in range(len(cv_scores)):
        print "Fold {}:".format(i) 
        print "    log loss - {}".format(cv_scores[i][0])
        print "    accuracy - {}".format(cv_scores[i][1])


    run_test(trn_all_df, test_df, features_to_use)



# TO TRY:
#     add manager listing counts
#     add manager level count (we only have percentage now)
#     add average image size
#     add ryans features from images