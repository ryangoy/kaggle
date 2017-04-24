import datetime
import json
import matplotlib.pyplot as plt
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

# one hots categorical features in 'features_to_use'
def vectorize_categorical_features(trn_df, test_df, features_to_use):
    categorical = ['display_address', 'manager_id', 'building_id', 'street_address']
    for f in categorical:
        if f in features_to_use and trn_df[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(trn_df[f].values) + list(test_df[f].values))
            trn_df[f] = lbl.transform(list(trn_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))

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
    if 'image_predictions_low' in features_to_use:
        id_to_pred = {}
        for fold in range(5):
            image_pred_ids = np.load('predictions/cnn/kfold_{}_ids.npy'.format(fold))
            image_preds = np.load('predictions/cnn/kfold_{}_preds.npy'.format(fold))
            print image_preds.shape, image_pred_ids.shape
            # print image_preds[0], image_pred_ids[0][image_pred_ids[0].find('/')+1:]
            for i in range(len(image_preds)):
                # if i <= 10:
                    # print 'a', type(image_pred_ids[i][5:]), image_pred_ids[i][5:]
                # print trn_df.iloc[i]['photos'][0][29:]
                # exit(1)
                id_to_pred[image_pred_ids[i][image_pred_ids[i].find('/')+1:]] = image_preds[i]
        image_preds_trn = np.zeros((len(trn_df), 3))
        image_preds_test = np.zeros((len(test_df), 3))
        cry = 0
        not_cry = 0
        for i in range(len(trn_df)):
            images = trn_df.iloc[i]['photos']
            for img in images:
                if i <= 5:
                    print 'b', type(str(img[29:])), str(img[29:])
                if str(img[29:]) in id_to_pred:
                    image_preds_trn[i] += id_to_pred[str(img[29:])]
                    not_cry += 1
                else:
                    cry += 1
            image_preds_trn[i] /= sum(image_preds_trn[i])
        print cry, not_cry
        cry = 0
        not_cry = 0
        for i in range(len(test_df)):
            images = test_df.iloc[i]['photos']
            for img in images:
                if str(img[29:]) in id_to_pred:
                    image_preds_test[i] += id_to_pred[str(img[29:])]
                    not_cry += 1
                else:
                    cry += 1
            image_preds_test[i] /= sum(image_preds_test[i])
        print cry, not_cry
        trn_df['image_predictions_low'] = image_preds_trn[:,0]
        test_df['image_predictions_low'] = image_preds_test[:,0]
        trn_df['image_predictions_medium'] = image_preds_trn[:,1]
        test_df['image_predictions_medium'] = image_preds_test[:,1]
        trn_df['image_predictions_high'] = image_preds_trn[:,2]
        test_df['image_predictions_high'] = image_preds_test[:,2]
    # magic apparently
    image_date = pd.read_csv('listing_image_time.csv')
    image_date.columns = ['listing_id', 'image_time_stamp']
    image_date.loc[80240,'image_time_stamp'] = 1478129766
    image_date = image_date.fillna(image_date.mean())
    image_date['img_date']                  = pd.to_datetime(image_date['image_time_stamp'], unit='s')
    image_date['img_days_passed']           = (image_date['img_date'].max() - image_date['img_date']).astype('timedelta64[D]').astype(int)
    image_date['img_date_month']            = image_date['img_date'].dt.month
    image_date['img_date_week']             = image_date['img_date'].dt.week
    image_date['img_date_day']              = image_date['img_date'].dt.day
    image_date['img_date_dayofweek']        = image_date['img_date'].dt.dayofweek
    image_date['img_date_dayofyear']        = image_date['img_date'].dt.dayofyear
    image_date['img_date_hour']             = image_date['img_date'].dt.hour
    image_date['img_date_monthBeginMidEnd'] = image_date['img_date_day'].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)
    trn_df = pd.merge(trn_df, image_date, on='listing_id', how='left')
    test_df = pd.merge(test_df, image_date, on='listing_id', how='left')

    return trn_df, test_df

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