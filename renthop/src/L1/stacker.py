import numpy as np
import pandas as pd
import paths
import xgboost as xgb
from os.path import join
import os

save_path = '/home/ryan/Desktop/submission.csv'

rf_paths = ['rf/L0_rf_fold1_0.57133413684.csv', 
             'rf/L0_rf_fold2_0.57024551903.csv',
             'rf/L0_rf_fold3_0.571273484183.csv',
             'rf/L0_rf_fold4_0.566444413366.csv',
             'rf/L0_rf_fold5_0.563739461319.csv',]
xgb_paths = ['xgb/L0_xgb_fold1_0.517224062756.csv',
            'xgb/L0_xgb_fold2_0.515606417015.csv',
            'xgb/L0_xgb_fold3_0.524019228236.csv',
            'xgb/L0_xgb_fold4_0.511632088679.csv',
            'xgb/L0_xgb_fold5_0.509369739084.csv', ]

cnn_path = 'cnn/cnn_train.csv'

# test_paths = ['cnn/cnn_test.csv', 'rf/L0_rf_magic_0.5686074029.csv', 'xgb/L0_xgb_magic_0.515570307154.csv', 
# ]
test_paths = ['rf/L0_rf_magic_0.5686074029.csv', 'xgb/L0_xgb_magic_0.515570307154.csv', 
]


def load_data(feature_paths):

    # put rf and xgb into their own respective dataframes (combine kfolds)
    features = []
    for f in range(len(feature_paths)):
        X_trn = pd.read_csv(join(paths.PREDICTIONS_DIR, feature_paths[f][0]))
        for i in range(1,len(feature_paths[f])):
            temp = pd.read_csv(join(paths.PREDICTIONS_DIR, feature_paths[f][i]))
            X_trn = pd.concat([X_trn, temp])
        features.append(X_trn)

    #load cnn preds
    #temp = pd.read_csv(join(paths.PREDICTIONS_DIR, cnn_path))
    #features.append(temp.drop('Unnamed: 0',1))

    # join on listing_id
    trn_df = pd.read_json(paths.TRAIN_JSON)
    X_trn = features[0]
    for i in range(1, len(features)):
        X_trn = X_trn.join(features[i].set_index('listing_id'), on='listing_id', how='inner', rsuffix='_'+str(i), sort=False)
    target_num_map = {'low':0, 'medium':1, 'high':2}
    trn_df['y'] = trn_df['interest_level'].apply(lambda x: target_num_map[x])
    y_trn = trn_df[trn_df['listing_id'].isin(X_trn['listing_id'])]

    # add labels corresponding to listing_id
    X_trn = X_trn.join(y_trn[['listing_id', 'y']].set_index('listing_id'), on='listing_id', how='inner', rsuffix='_y')

    # assemble test data
    X_test = pd.read_csv(join(paths.PREDICTIONS_DIR, test_paths[0]))
    #X_test = X_test.drop('Unnamed: 0',1)
    for i in range(1,len(test_paths)):
        temp = pd.read_csv(join(paths.PREDICTIONS_DIR, test_paths[i]))
        X_test = X_test.join(temp.set_index('listing_id'), on='listing_id', how='outer', rsuffix='_'+str(i), sort=False)

    return X_trn.drop(['listing_id', 'y'], 1), X_trn['y'], X_test, None, 


def run_stacker(X_trn, y_trn, X_test, y_test=None, seed_val=0, num_rounds=100):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.3
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = 'mlogloss'
    param['min_child_weight'] = 1
    param['subsample'] = .7
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
        xgtest = xgb.DMatrix(X_test.drop('listing_id', 1))
        model = xgb.train(plst, xgtrain, num_rounds, verbose_eval = True)
        #model = xgb.cv(plst, xgtrain, num_boost_round=num_rounds, nfold=5, verbose_eval=True,
        #               seed=0)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

if __name__ == '__main__':
    X_trn, y_trn, X_test, y_test = load_data([xgb_paths, rf_paths])
    preds, model = run_stacker(X_trn, y_trn, X_test, y_test)
    preds = pd.DataFrame(preds).clip(lower=0.001)
    preds = preds.div(preds.sum(axis=1), axis=0)
    ret = pd.concat([X_test['listing_id'],pd.DataFrame(preds)],  axis=1)
    ret=ret.reindex(columns=['listing_id', 2, 1,0])
    
    ret.to_csv(save_path)

