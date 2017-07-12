import numpy as np
import pandas as pd
import lightgbm as lgb
from model import Model

NUM_ROUND_SPLIT = .8
EARLY_STOPPING_ROUNDS = 20
VERBOSE_INTERVAL = False
NUM_BOOST_ROUND = 1000

class LightGBM(Model):
    """
    Basic XGB model.
    """
    def __init__(self, lgb_params=None, log_data=True, name='LightGBM', features=None):
        self.model = None
        self.num_boost_rounds = None
        self.log_data = True
        if lgb_params is None:
            self.lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2', 'rmse'},
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'max_depth': 60,

            }
        else:
            self.lgb_params = lgb_params
        self.name = name
        self.features = features

    def find_num_boost_round(self, X_trn, y_trn):
        """
        Find the number of iterations we should run using a mini
        validation set. This way we don't overfit the predictions.
        """
        split = int(NUM_ROUND_SPLIT * X_trn.shape[0])
        d_trn = lgb.Dataset(X_trn.iloc[:split], label=y_trn.iloc[:split])
        d_val = lgb.Dataset(X_trn.iloc[split:], label=y_trn.iloc[split:], 
                            reference=d_trn)
        val_model = lgb.train(self.lgb_params, d_trn, 
                  num_boost_round=NUM_BOOST_ROUND,
                  valid_sets=d_val,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                  verbose_eval=False)
        return val_model.best_iteration

    def train(self, X_trn, y_trn, X_val=None, y_val=None):
        if self.features is not None:
            X_trn = X_trn[self.features]
            if X_val is not None:
                X_val = X_val[self.features]
        if self.log_data:
            y_trn = np.log1p(y_trn)
        if self.num_boost_rounds is None:
            self.num_boost_rounds = self.find_num_boost_round(X_trn, y_trn)
        self.lgb_params['task'] = 'train'
        lgb_train = lgb.Dataset(X_trn, y_trn)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        self.model = lgb.train(self.lgb_params, lgb_train, 
            num_boost_round=self.num_boost_rounds, verbose_eval=False,
            valid_sets=lgb_val)

    def test(self, X_test, y_test=None):
        if self.features is not None:
            X_test = X_test[self.features]
        self.lgb_params['task'] = 'prediction'
        preds = self.model.predict(X_test, num_iteration=self.num_boost_rounds)
        if self.log_data:
            preds = np.exp(preds) - 1
        return preds