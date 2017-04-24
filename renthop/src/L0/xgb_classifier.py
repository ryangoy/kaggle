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

from feature_extraction import vectorize_categorical_features, feature_engineering, cv_stats2

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


def run_validation(trn_df, val_df, features_to_use):
    start_time = time.time()
    print '[START] validation run'
    
    # trn_df = trn_df[trn_df.bathrooms != 0]
    # trn_df.is_copy = False

    vectorize_categorical_features(trn_df, val_df, features_to_use)
    print '[TIME] to vectorize categorical features:', time.time() - start_time

    trn_df, val_df = feature_engineering(trn_df, val_df, features_to_use)
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

    print trn_df[features_to_use].shape

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

    conf_mat = confusion_matrix(y_val, np.argmax(preds, axis=1))
    print 'confusion matrix:\n', conf_mat
    print 'percent confusion matrix:\n', (conf_mat+0.0)/np.sum(conf_mat)
    print 'log loss:', fold_log_loss
    print 'accuracy:', fold_accuracy

    out_df = pd.DataFrame(preds)
    out_df.columns = ['low', 'medium', 'high']
    out_df['listing_id'] = val_df.listing_id.values

    print '[FINISH] validation run'

    return fold_log_loss, fold_accuracy, preds, out_df


def run_test(trn_df, test_df, features_to_use):
    start_time = time.time()
    print '[START] test run'

    vectorize_categorical_features(trn_df, test_df, features_to_use)
    print '[TIME] to vectorize categorical features:', time.time() - start_time

    trn_df, test_df = feature_engineering(trn_df, test_df, features_to_use)
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

    print trn_df[features_to_use].shape
    
    X_trn = sparse.hstack([trn_df[features_to_use], trn_sparse]).tocsr()
    X_test = sparse.hstack([test_df[features_to_use], test_sparse]).tocsr()

    target_num_map = {'low':0, 'medium':1, 'high':2}
    y_trn = np.array(trn_df['interest_level'].apply(lambda x: target_num_map[x]))
    print X_trn.shape, X_test.shape, y_trn.shape

    print '[TIME] to create train/validation matrices:', time.time() - start_time

    preds, model = runXGB(X_trn, y_trn, X_test, num_rounds=800)

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
                        # 'created_percent', 
                        # 'created_month',
                        # 'created_day',
                        'created_hour',
                        'density',
                        # 'average_image_size',
                        # 'average_image_width',
                        # 'average_image_height',
                        # 'average_image_diagonal',
                        # 'image_predictions_low', 'image_predictions_medium', 'image_predictions_high', 
                        'image_time_stamp',
                        # 'img_days_passed',
                        # 'img_date_month',
                        # 'img_date_week',
                        'img_date_day',
                        'img_date_dayofweek',
                        # 'img_date_dayofyear',
                        'img_date_hour',
                        # 'img_date_monthBeginMidEnd',
                        # cv stats or whatever that means
                        'manager_level_percent_low', 'manager_level_percent_medium', 'manager_level_percent_high',
                        # 'manager_level_count_low', 'manager_level_count_medium', 'manager_level_count_high',
                        # 'manager_listings_count',
                        # 'manager_skill',
                      ]

    # trn_stacker=[ [0.0 for s in range(3)]  for k in range (0,(trn_all_df.shape[0])) ]
    # test_stacker=[[0.0 for s in range(3)]   for k in range (0,(test_df.shape[0]))]

    cv_scores = []
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
    for trn_index, val_index in kf.split(range(trn_all_df.shape[0])):
        # break
        print 'VALIDATION FOLD', (len(cv_scores) + 1)
        trn_df = trn_all_df.iloc[trn_index]
        val_df = trn_all_df.iloc[val_index]
        trn_df.is_copy = False
        val_df.is_copy = False
        fold_log_loss, fold_accuracy, preds, out_df = run_validation(trn_df, val_df, features_to_use)
        cv_scores += [[fold_log_loss, fold_accuracy]]

        out_df.to_csv('predictions/L0_xgb_fold{}_{}.csv'.format(len(cv_scores), fold_log_loss), index=False)
        # break
        # no=0
        # for real_index in val_index:
        #     for d in range (0,3):
        #         trn_stacker[real_index][d]=(preds[no][d])
        #     no+=1
        # break
    # exit(1)
    if len(cv_scores) > 1:
        for i in range(len(cv_scores)):
            print 'Fold {}:'.format(i+1) 
            print '    log loss - {}'.format(cv_scores[i][0])
            print '    accuracy - {}'.format(cv_scores[i][1])


    X_trn, X_test, y_trn, preds, out_df = run_test(trn_all_df, test_df, features_to_use)
    mean_log_loss = sum([c[0] for c in cv_scores])/len(cv_scores)
    out_df.to_csv('predictions/L0_xgb_magic_{}.csv'.format(mean_log_loss), index=False)
    # out_df.to_csv('predictions/L0_xgb_magic.csv', index=False)


    # for pr in range (0,len(preds)):  
    #     for d in range (0,3):            
    #         test_stacker[pr][d]=(preds[pr][d]) 

    # trn_file ='train_stacknet.csv'
    # test_file ='test_stacknet.csv'

    # ids = test_df.listing_id.values

    # print ('merging columns')   
    # #stack xgboost predictions
    # X_trn=np.column_stack((X_trn.toarray(),trn_stacker))
    # # stack id to test
    # X_test=np.column_stack((X_test.toarray(),test_stacker))        
    
    # # stack target to train
    # X_trn=np.column_stack((y_trn,X_trn))
    # # stack id to test
    # X_test=np.column_stack((ids,X_test))
    
    # #export to txt files (, del.)
    # print ('exporting files')
    # np.savetxt(trn_file, X_trn, delimiter=',', fmt='%.5f')
    # np.savetxt(test_file, X_test, delimiter=',', fmt='%.5f')        

    # mean_log_loss = 

    # print('Write results...')
    # output_file = 'submission_'+str(mean_log_loss)+'.csv'
    # print('Writing submission to %s' % output_file)
    # f = open(output_file, 'w')   
    # f.write('listing_id,high,medium,low\n')# the header   
    # for g in range(0, len(test_stacker))  :
    #   f.write('%s' % (ids[g]))
    #   for prediction in test_stacker[g]:
    #      f.write(',%f' % (prediction))    
    #   f.write('\n')
    # f.close()
    # print('Done.')



# TO TRY:
#     add average image size
#     add ryans features from images