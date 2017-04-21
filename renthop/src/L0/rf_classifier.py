import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy import sparse
import sklearn
from sklearn import model_selection, preprocessing, ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import time

random.seed(0)
np.random.seed(0)

# runs sklearn random forest for validation and test runs
def runRF(X_trn, y_trn, X_test, y_test=None, feature_names=None, seed_val=0, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', max_depth=15, 
                                   min_samples_split=10, min_samples_leaf=1, 
                                   min_weight_fraction_leaf=0.0, max_features='auto', 
                                   max_leaf_nodes=None, min_impurity_split=1e-07, 
                                   bootstrap=True, n_jobs=3, random_state=seed_val, verbose=1)

    # model = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=10, min_samples_split=5, 
    #                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
    #                              max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=False, 
    #                              n_jobs=3, random_state=seed_val, verbose=1)

    # dt = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=5, 
    #                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
    #                             random_state=seed_val, max_leaf_nodes=None, min_impurity_split=1e-07)
    # model = AdaBoostClassifier(base_estimator=dt, n_estimators=n_estimators, learning_rate=1.0, 
    #                            algorithm='SAMME.R', random_state=seed_val)

    # model = LogisticRegression()

    # model = MLPClassifier()


    model.fit(X_trn, y_trn)

    pred_test_y = model.predict_proba(X_test)
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
        trn_df['price_t'] = trn_df['price']/np.maximum(trn_df['bedrooms'], 1)
        test_df['price_t'] = test_df['price']/np.maximum(test_df['bedrooms'], 1)
    # price per bathroom
    if 'price_s' in features_to_use:
        trn_df['price_s'] = trn_df['price']/np.maximum(trn_df['bathrooms'], 1)
        test_df['price_s'] = test_df['price']/np.maximum(test_df['bathrooms'], 1)
    # price per room
    if 'price_r' in features_to_use:
        trn_df['price_r'] = trn_df['price']/np.maximum(trn_df['rooms'], 1)
        test_df['price_r'] = test_df['price']/np.maximum(test_df['rooms'], 1)
    if 'log_price' in features_to_use:
        trn_df['log_price'] = np.log(trn_df['price'])
        test_df['log_price'] = np.log(test_df['price'])
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
    # day at which listing was created
    if 'created_month' in features_to_use:
        def created_month(date_unicode):
            date = datetime.datetime.strptime(date_unicode, '%Y-%m-%d %H:%M:%S')
            return date.month
        trn_df['created_month'] = trn_df['created'].apply(lambda x: created_month(x))
        test_df['created_month'] = test_df['created'].apply(lambda x: created_month(x))
    # day at which listing was created
    if 'created_day' in features_to_use:
        def created_day(date_unicode):
            date = datetime.datetime.strptime(date_unicode, '%Y-%m-%d %H:%M:%S')
            return date.day
        trn_df['created_day'] = trn_df['created'].apply(lambda x: created_day(x))
        test_df['created_day'] = test_df['created'].apply(lambda x: created_day(x))
    # hour at which listing was created
    if 'created_hour' in features_to_use:
        def created_hour(date_unicode):
            date = datetime.datetime.strptime(date_unicode, '%Y-%m-%d %H:%M:%S')
            return date.hour
        trn_df['created_hour'] = trn_df['created'].apply(lambda x: created_hour(x))
        test_df['created_hour'] = test_df['created'].apply(lambda x: created_hour(x))
    if 'density' in features_to_use:
        trn_df['pos'] = trn_df.longitude.round(3).astype(str) + '_' + trn_df.latitude.round(3).astype(str)
        test_df['pos'] = test_df.longitude.round(3).astype(str) + '_' + test_df.latitude.round(3).astype(str)
        vals = trn_df['pos'].value_counts()
        dvals = vals.to_dict()
        trn_df['density'] = trn_df['pos'].apply(lambda x: dvals.get(x, vals.min()))
        test_df['density'] = test_df['pos'].apply(lambda x: dvals.get(x, vals.min()))
    # average image size, wight, height
    image_info = np.load('data/img_stats.npy')
    id_to_img = {}
    for i in range(image_info.shape[0]):
        id_to_img[image_info[i, 0]] = image_info[i, 1:]
    img_size_trn = [np.nan]*len(trn_df)
    img_width_trn = [np.nan]*len(trn_df)
    img_height_trn = [np.nan]*len(trn_df)
    img_diag_trn = [np.nan]*len(trn_df)
    img_size_test = [np.nan]*len(test_df)
    img_width_test = [np.nan]*len(test_df)
    img_height_test = [np.nan]*len(test_df)
    img_diag_test = [np.nan]*len(test_df)
    for i in range(trn_df.shape[0]):
        if trn_df.iloc[i]['listing_id'] not in id_to_img:
            continue
        img_size_trn[i] = id_to_img[trn_df.iloc[i]['listing_id']][2]
        img_width_trn[i] = id_to_img[trn_df.iloc[i]['listing_id']][0]
        img_height_trn[i] = id_to_img[trn_df.iloc[i]['listing_id']][1]
        img_diag_trn[i] = id_to_img[trn_df.iloc[i]['listing_id']][3]
    for i in range(test_df.shape[0]):
        if test_df.iloc[i]['listing_id'] not in id_to_img:
            continue
        img_size_test[i] = id_to_img[test_df.iloc[i]['listing_id']][2]
        img_width_test[i] = id_to_img[test_df.iloc[i]['listing_id']][0]
        img_height_test[i] = id_to_img[test_df.iloc[i]['listing_id']][1]
        img_diag_test[i] = id_to_img[test_df.iloc[i]['listing_id']][3]
    if 'average_image_size' in features_to_use:
        trn_df['average_image_size'] = img_size_trn
        test_df['average_image_size'] = img_size_test
    if 'average_image_width' in features_to_use:
        trn_df['average_image_width'] = img_width_trn
        test_df['average_image_width'] = img_width_test
    if 'average_image_height' in features_to_use:
        trn_df['average_image_height'] = img_height_trn
        test_df['average_image_height'] = img_height_test
    if 'average_image_diagonal' in features_to_use:
        trn_df['average_image_diagonal'] = img_diag_trn
        test_df['average_image_diagonal'] = img_diag_test
    # cnn image predictions
    # if 'image_predictions_low' in features_to_use:
    #     image_preds = np.load('predictions/cnn_preds.npy')
    #     image_pred_ids = np.load('predictions/cnn_pred_ids.npy')
    #     id_to_pred = {}
    #     for i in range(len(image_preds)):
    #         if i <= 10:
    #             print 'a', type(image_pred_ids[i][5:]), image_pred_ids[i][5:]
    #         # print trn_df.iloc[i]['photos'][0][29:]
    #         # exit(1)
    #         id_to_pred[image_pred_ids[i][5:]] = image_preds[i]
    #     image_preds_trn = np.zeros((len(trn_df), 3))
    #     image_preds_test = np.zeros((len(test_df), 3))
    #     cry = 0
    #     not_cry = 0
    #     for i in range(len(trn_df)):
    #         images = trn_df.iloc[i]['photos']
    #         for img in images:
    #             if i <= 5:
    #                 print 'b', type(str(img[29:])), str(img[29:])
    #             if str(img[29:]) in id_to_pred:
    #                 image_preds_trn[i] += id_to_pred[str(img[29:])]
    #                 not_cry += 1
    #             else:
    #                 cry += 1
    #         image_preds_trn[i] /= sum(image_preds_trn[i])
    #     print cry, not_cry
    #     cry = 0
    #     not_cry = 0
    #     for i in range(len(test_df)):
    #         images = test_df.iloc[i]['photos']
    #         for img in images:
    #             if str(img[29:]) in id_to_pred:
    #                 image_preds_test[i] += id_to_pred[str(img[29:])]
    #                 not_cry += 1
    #             else:
    #                 cry += 1
    #         image_preds_test[i] /= sum(image_preds_test[i])
    #     print cry, not_cry
    #     trn_df['image_predictions_low'] = image_preds_trn[:,0]
    #     test_df['image_predictions_low'] = image_preds_test[:,0]
    #     trn_df['image_predictions_medium'] = image_preds_trn[:,1]
    #     test_df['image_predictions_medium'] = image_preds_test[:,1]
    #     trn_df['image_predictions_high'] = image_preds_trn[:,2]
    #     test_df['image_predictions_high'] = image_preds_test[:,2]

# my version of cv stats - overfits horridly since there is too much information on manager for sparse managers
def cv_stats(trn_df, test_df, features_to_use):
    l_trn = [[np.nan]*len(trn_df) for _ in range(2)]
    m_trn = [[np.nan]*len(trn_df) for _ in range(2)]
    h_trn = [[np.nan]*len(trn_df) for _ in range(2)]
    count_trn = [np.nan]*len(trn_df)

    l_test = [[np.nan]*len(test_df) for _ in range(2)]
    m_test = [[np.nan]*len(test_df) for _ in range(2)]
    h_test = [[np.nan]*len(test_df) for _ in range(2)]
    count_test = [np.nan]*len(test_df)

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
        if sum(levels) >= 50:
            l_trn[0][i] = levels[0]*1.0/sum(levels)
            m_trn[0][i] = levels[1]*1.0/sum(levels)
            h_trn[0][i] = levels[2]*1.0/sum(levels)
            l_trn[1][i] = levels[0]*1.0
            m_trn[1][i] = levels[1]*1.0
            h_trn[1][i] = levels[2]*1.0
            count_trn[i] = sum(levels)
        levels[index] += 1
    for i in range(len(test_df)):
        temp = test_df.iloc[i]
        if temp['manager_id'] not in manager_level.keys() or sum(manager_level[temp['manager_id']]) == 0:
            l_test[0][i] = np.nan
            m_test[0][i] = np.nan
            h_test[0][i] = np.nan
            l_test[1][i] = np.nan
            m_test[1][i] = np.nan
            h_test[1][i] = np.nan
            count_test[i] = 0
        else:
            levels = manager_level[temp['manager_id']]
            l_test[0][i] = levels[0]*1.0/sum(levels)
            m_test[0][i] = levels[1]*1.0/sum(levels)
            h_test[0][i] = levels[2]*1.0/sum(levels)
            l_test[1][i] = levels[0]*1.0
            m_test[1][i] = levels[1]*1.0
            h_test[1][i] = levels[2]*1.0
            count_test[i] = sum(levels)

    if 'manager_level_percent_low' in features_to_use:
        trn_df['manager_level_percent_low'] = l_trn[0]
        trn_df['manager_level_percent_medium'] = m_trn[0]
        trn_df['manager_level_percent_high'] = h_trn[0]
        test_df['manager_level_percent_low'] = l_test[0]
        test_df['manager_level_percent_medium'] = m_test[0]
        test_df['manager_level_percent_high'] = h_test[0]
    if 'manager_level_count_low' in features_to_use:
        trn_df['manager_level_count_low'] = l_trn[1]
        trn_df['manager_level_count_medium'] = m_trn[1]
        trn_df['manager_level_count_high'] = h_trn[1]
        test_df['manager_level_count_low'] = l_test[1]
        test_df['manager_level_count_medium'] = m_test[1]
        test_df['manager_level_count_high'] = h_test[1]
    if 'manager_listings_count' in features_to_use:
        trn_df['manager_listings_count'] = count_trn
        test_df['manager_listings_count'] = count_test

    # histogram of listings per manager
    # sums = []
    # for manager in manager_level:
    #     # print sum(manager_level[manager])
    #     if sum(manager_level[manager]) > 0:
    #     #     print manager_level[manager]
    #         sums += [sum(manager_level[manager])]
    # plt.hist(sums, bins=np.logspace(0.0, 3.5, 35))
    # plt.xscale('log')
    # plt.show()
    # exit(1)

# original version of cv stats, better as it uses only part of data to validate against each sample
def cv_stats2(trn_df, test_df, features_to_use, splits=5):
    indicies = list(range(trn_df.shape[0]))
    random.shuffle(indicies)
    l_trn = [[.7]*len(trn_df) for _ in range(2)]
    m_trn = [[.22]*len(trn_df) for _ in range(2)]
    h_trn = [[.08]*len(trn_df) for _ in range(2)]
    count_trn = [0]*len(trn_df)
    skill_trn = [.38]*len(trn_df)

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
            if sum(levels) != 0:
                l_trn[0][j] = levels[0]*1.0/sum(levels)
                m_trn[0][j] = levels[1]*1.0/sum(levels)
                h_trn[0][j] = levels[2]*1.0/sum(levels)
                l_trn[1][j] = levels[0]*1.0
                m_trn[1][j] = levels[1]*1.0
                h_trn[1][j] = levels[2]*1.0
                count_trn[j] = sum(levels)
                skill_trn[j] = (levels[1]*1.0+levels[2]*2.0)/sum(levels)

    l_test = [[.7]*len(test_df) for _ in range(2)]
    m_test = [[.22]*len(test_df) for _ in range(2)]
    h_test = [[.08]*len(test_df) for _ in range(2)]
    count_test = [0]*len(test_df)
    skill_test = [.38]*len(test_df)

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
        if temp['manager_id'] in manager_level.keys():
            levels = manager_level[temp['manager_id']]
            l_test[0][j] = levels[0]*1.0/sum(levels)
            m_test[0][j] = levels[1]*1.0/sum(levels)
            h_test[0][j] = levels[2]*1.0/sum(levels)
            l_test[1][j] = levels[0]*1.0
            m_test[1][j] = levels[1]*1.0
            h_test[1][j] = levels[2]*1.0
            count_test[j] = sum(levels)
            skill_test[j] = (levels[1]*1.0+levels[2]*2.0)/sum(levels)

    if 'manager_level_percent_low' in features_to_use:
        trn_df['manager_level_percent_low'] = l_trn[0]
        trn_df['manager_level_percent_medium'] = m_trn[0]
        trn_df['manager_level_percent_high'] = h_trn[0]
        test_df['manager_level_percent_low'] = l_test[0]
        test_df['manager_level_percent_medium'] = m_test[0]
        test_df['manager_level_percent_high'] = h_test[0]
    if 'manager_level_count_low' in features_to_use:
        trn_df['manager_level_count_low'] = l_trn[1]
        trn_df['manager_level_count_medium'] = m_trn[1]
        trn_df['manager_level_count_high'] = h_trn[1]
        test_df['manager_level_count_low'] = l_test[1]
        test_df['manager_level_count_medium'] = m_test[1]
        test_df['manager_level_count_high'] = h_test[1]
    if 'manager_listings_count' in features_to_use:
        trn_df['manager_listings_count'] = count_trn
        test_df['manager_listings_count'] = count_test
    if 'manager_skill' in features_to_use:
        trn_df['manager_skill'] = skill_trn
        test_df['manager_skill'] = skill_test


def run_validation(trn_df, val_df, features_to_use):
    start_time = time.time()
    print '[START] validation run'
    
    # trn_df = trn_df[trn_df.bathrooms != 0]
    # trn_df.is_copy = False

    vectorize_categorical_features(trn_df, val_df, features_to_use)
    print '[TIME] to vectorize categorical features:', time.time() - start_time

    feature_engineering(trn_df, val_df, features_to_use)
    print '[TIME] to engineer features:', time.time() - start_time

    # cv_stats(trn_df, val_df, features_to_use)
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

    preds, model = runRF(X_trn, y_trn, X_val, y_val)

    print '[TIME] to run sklearn random forest:', time.time() - start_time

    fold_log_loss = log_loss(y_val, preds)
    fold_accuracy = accuracy_score(y_val, np.argmax(preds, axis=1))

    conf_mat = confusion_matrix(y_val, np.argmax(preds, axis=1))
    print 'confusion matrix:\n', conf_mat
    print 'percent confusion matrix:\n', (conf_mat+0.0)/np.sum(conf_mat)
    print 'log loss:', fold_log_loss
    print 'accuracy:', fold_accuracy

    print '[FINISH] validation run'

    return fold_log_loss, fold_accuracy, preds


def run_test(trn_df, test_df, features_to_use):
    start_time = time.time()
    print '[START] test run'

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

    preds, model = runRF(X_trn, y_trn, X_test, n_estimators=100)

    print '[TIME] to run xgboost:', time.time() - start_time

    out_df = pd.DataFrame(preds)
    out_df.columns = ['low', 'medium', 'high']
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
    out_df['listing_id'] = test_df.listing_id.values
    # out_df.to_csv('xgb_basic_cv_stat.csv', index=False)

    print '[TIME] to created submission:', time.time() - start_time

    print '[FINISH] test run'

    return X_trn, X_test, y_trn, preds, out_df


if __name__ == '__main__':
    trn_all_df = pd.read_json('data/train.json')
    test_df = pd.read_json('data/test.json')

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
                        'log_price', 
                        'num_photos', 
                        'num_features', 
                        'num_description_words',
                        # 'created_year_percent', 
                        'created_percent', 
                        'created_month',
                        'created_day',
                        'created_hour',
                        'density',
                        # 'average_image_size',
                        # 'average_image_width',
                        # 'average_image_height',
                        # 'average_image_diagonal',
                        # 'image_predictions_low', 'image_predictions_medium', 'image_predictions_high', 
                        # cv stats or whatever that means
                        'manager_level_percent_low', 'manager_level_percent_medium', 'manager_level_percent_high',
                        'manager_level_count_low', 'manager_level_count_medium', 'manager_level_count_high',
                        'manager_listings_count',
                        'manager_skill',
                      ]

    cv_scores = []
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for trn_index, val_index in kf.split(range(trn_all_df.shape[0])):
        print 'VALIDATION FOLD', (len(cv_scores) + 1)
        trn_df = trn_all_df.iloc[trn_index]
        val_df = trn_all_df.iloc[val_index]
        trn_df.is_copy = False
        val_df.is_copy = False
        fold_log_loss, fold_accuracy, preds = run_validation(trn_df, val_df, features_to_use)
        cv_scores += [[fold_log_loss, fold_accuracy]]

    if len(cv_scores) > 1:
        for i in range(len(cv_scores)):
            print 'Fold {}:'.format(i+1) 
            print '    log loss - {}'.format(cv_scores[i][0])
            print '    accuracy - {}'.format(cv_scores[i][1])


    X_trn, X_test, y_trn, preds, out_df = run_test(trn_all_df, test_df, features_to_use)
    mean_log_loss = sum([c[0] for c in cv_scores])/len(cv_scores)
    out_df.to_csv('predictions/L0_xgb_{}.csv'.format(mean_log_loss), index=False)
