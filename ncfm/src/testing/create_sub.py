import pandas as pd
import numpy as np
from os.path import join
import os, sys
import xgboost as xgb

ds_path = '/home/ryan/cs/datasets/ncfm/blended'

stage1_csv = join(ds_path, 'preds1.csv')

stage2_preds = np.load(join(ds_path, 'FINAL_PREDS.npy'))
stage2_conf =  np.load(join(ds_path, 'FINAL_CONF.npy'))

fish_preds = np.load(join(ds_path, 'fish_train_preds.npy'))
fish_labels = np.load(join(ds_path, 'fish_train_labels.npy'))
fish_conf = np.load(join(ds_path, 'fish_train_conf.npy'))
nof_preds = np.load(join(ds_path, 'nof_train_preds.npy'))
nof_conf = np.load(join(ds_path, 'nof_train_conf.npy'))

stage2_images = join(ds_path, '../test_stg2')

num_nof = 80

labels = np.zeros((fish_labels.shape[0] + num_nof, 8))
labels[:fish_labels.shape[0],: -1] = fish_labels
labels[fish_labels.shape[0]:, -1] = 1

train = np.zeros((fish_preds.shape[0] + num_nof, 8))
train[:fish_preds.shape[0],: -1] = fish_preds
train[:fish_preds.shape[0],  -1] = fish_conf
train[fish_preds.shape[0]:, :-1] = nof_preds[:num_nof]
train[fish_preds.shape[0]:, -1] = nof_conf[:num_nof]


test = np.zeros((stage2_preds.shape[0], 8))
test[:, :-1] = stage2_preds
test[:, -1] = stage2_conf

# test = test[:10]


param = {}
param['objective'] = 'multi:softprob'
param['eta'] = .1
# param['max_depth'] = 10
param['silent'] = 1
param['num_class'] = 8
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
# param['seed'] = 0
num_rounds = 10000
plst = list(param.items())
xg_trn = xgb.DMatrix(train, label=np.argmax(labels, axis=1))
xg_tst = xgb.DMatrix(test)

watchlist = [ (xg_trn,'train'),  ]
model = xgb.train(plst, xg_trn, num_rounds, watchlist, early_stopping_rounds=30)

silverpls = model.predict(xg_tst)

old_df = pd.DataFrame.from_csv(stage1_csv)


image_names = sorted(os.listdir(stage2_images))
image_names = ['test_stg2/'+ i for i in image_names]

df = pd.DataFrame(columns=old_df.columns, index=image_names)

for i in range(len(image_names)):
    df.loc[image_names[i]] = [silverpls[i, 0], silverpls[i, 1], silverpls[i, 2], silverpls[i, 3], 
                                 silverpls[i, 7], silverpls[i, 4], silverpls[i, 5], silverpls[i, 6]]

final = old_df.append(df)
final = final.clip(0.01, 1.0)
final = final.div(final.sum(axis=1), axis=0)

final.to_csv("/home/ryan/Desktop/ssd_plus_vgg16.csv")

