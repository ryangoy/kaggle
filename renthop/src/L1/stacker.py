import numpy as np
import pandas as pd
import paths
import sys
import xgboost as xgb
from os.path import join
import os
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import log_loss

save_path = paths.FINAL_PREDICTION

NUM_ROUNDS = 45

xgb_paths = ['xgb/L0_xgb_fold1_0.517224062756.csv',
            'xgb/L0_xgb_fold2_0.515606417015.csv',
            'xgb/L0_xgb_fold3_0.524019228236.csv',
            'xgb/L0_xgb_fold4_0.511632088679.csv',
            'xgb/L0_xgb_fold5_0.509369739084.csv', ]

rf_paths = ['rf/L0_rf_fold1_0.57133413684.csv', 
             'rf/L0_rf_fold2_0.57024551903.csv',
             'rf/L0_rf_fold3_0.571273484183.csv',
             'rf/L0_rf_fold4_0.566444413366.csv',
             'rf/L0_rf_fold5_0.563739461319.csv',]


cnn_paths = ['cnn/kfold_0.csv',
             'cnn/kfold_1.csv',
             'cnn/kfold_2.csv',
             'cnn/kfold_3.csv',
             'cnn/kfold_4.csv',]

lin_paths = ['L0_lin_fold1_10.2607742113.csv',
               'L0_lin_fold2_10.3450393952.csv',
                'L0_lin_fold3_10.2038377471.csv',
                'L0_lin_fold4_10.288596339.csv',
                'L0_lin_fold5_10.1658406557.csv']

test_paths = [ 'xgb/L0_xgb_magic_0.515570307154.csv', 'rf/L0_rf_magic_0.5686074029.csv', 
'cnn/cnn_test.csv', 'L0_lin_magic_10.2528176697.csv']

# order is very important cause i just load it in into the dataframe in this order
#test_paths = ['xgb/L0_xgb_magic_0.515570307154.csv','rf/L0_rf_magic_0.5686074029.csv', ]

extra_features = ['listing_id', 'bedrooms', 'price']


def load_data(feature_paths, test_paths):

    # put rf and xgb into their own respective dataframes (combine kfolds)
    features = []
    trn_json = pd.read_json(paths.TRAIN_JSON)
    # loop through folds to create each combined fold
    for fold in range(len(feature_paths[0])):
        X_trn = pd.read_csv(join(paths.PREDICTIONS_DIR, feature_paths[0][fold]))
        #look through each L0 classifier we have
        for i in range(1,len(feature_paths)):
            temp = pd.read_csv(join(paths.PREDICTIONS_DIR, feature_paths[i][fold]))
            X_trn = X_trn.join(temp.set_index('listing_id'), on='listing_id', how='inner', rsuffix='_'+str(i), sort=False)          
        features.append(X_trn)

    target_num_map = {'low':0, 'medium':1, 'high':2}
    trn_json['y'] = trn_json['interest_level'].apply(lambda x: target_num_map[x])

    # add labels corresponding to listing_id
    labeled_features = []
    for X_trn in features:
        X_trn = X_trn.join(trn_json[extra_features + ['y']].set_index('listing_id'), on='listing_id', how='left')
        labeled_features.append(X_trn)

    #     print log_loss(X_trn['y'], X_trn[['low_2', 'medium_2', 'high_2']]
    #         .fillna(X_trn[['low_2', 'medium_2', 'high_2']].mean()), )
    #     print X_trn[['low_2', 'medium_2', 'high_2']].head()
    # sys.exit()


    # assemble test data
    test_json = pd.read_json(paths.TEST_JSON)[extra_features]
    X_test = pd.read_csv(join(paths.PREDICTIONS_DIR, test_paths[0]))
    for i in range(1,len(test_paths)):
        temp = pd.read_csv(join(paths.PREDICTIONS_DIR, test_paths[i]))
        X_test = X_test.join(temp.set_index('listing_id'), on='listing_id', how='inner', rsuffix='_'+str(i), sort=False)
    X_test = X_test.join(test_json.set_index('listing_id'), on='listing_id', how='inner', rsuffix='_'+str(i), sort=False)
    return labeled_features, X_test


def run_xgb_stacker(features, X_test, num_rounds, seed_val=0):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.3
    param['max_depth'] = 2
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = 'mlogloss'
    param['min_child_weight'] = 1
    param['subsample'] = .9
    param['colsample_bytree'] = 0.9
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())

    for fold in range(len(rf_paths)):
        dfs = []
        for i in range(len(rf_paths)):
            if i == fold:
                continue
            else:
                dfs.append(features[i])
        fold_train = pd.concat(dfs)
        xgtrain = xgb.DMatrix(fold_train.drop('y',1), label=fold_train['y'])


        xgval = xgb.DMatrix(features[fold].drop('y',1), label=features[fold]['y'])
        watchlist = [ (xgtrain,'train'), (xgval, 'val') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10)

    fold_train = pd.concat(features)
    xgtrain = xgb.DMatrix(fold_train.drop(['y'],1), label=fold_train['y'])
    watchlist = [ (xgtrain,'train')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10)
    xgtest = xgb.DMatrix(X_test)
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

def run_logistic_stacker(features, X_test, seed_val=0):

    imputed_features = []
    for X in features:
        imputed_features.append(X.fillna(X.mean()))
    features = imputed_features

    for fold in range(len(rf_paths)):
        dfs = []
        for i in range(len(rf_paths)):
            if i == fold:
                continue
            else:
                dfs.append(features[i])
        fold_train = pd.concat(dfs)

        model = ElasticNet(random_state=0)
        model.fit(fold_train.drop(['listing_id', 'y'],1), fold_train['y'])
        fold_preds = model.predict_proba(features[fold].drop(['listing_id', 'y'],1))
        print "Loss for fold {} is {}.".format(fold+1, log_loss(features[fold]['y'], fold_preds))


    fold_train = pd.concat(features)
    model.fit(fold_train.drop(['listing_id', 'y'],1), fold_train['y'])
    pred_test_y = model.predict_proba(X_test.drop('listing_id',1))
    return pred_test_y, model



if __name__ == '__main__':
    features, X_test = load_data([xgb_paths, rf_paths, cnn_paths, lin_paths], test_paths)
    preds, model = run_xgb_stacker(features, X_test, num_rounds=NUM_ROUNDS)
    #preds, model = run_logistic_stacker(features, X_test)
    preds = pd.DataFrame(preds).clip(lower=0.001)
    preds = preds.div(preds.sum(axis=1), axis=0)
    ret = pd.concat([X_test['listing_id'],pd.DataFrame(preds)],  axis=1)
    ret=ret.reindex(columns=['listing_id', 2, 1,0])
    
    ret.to_csv(save_path)

