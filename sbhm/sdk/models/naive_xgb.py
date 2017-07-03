import numpy as np
import pandas as pd
import xgboost as xgb
from model import Model

NUM_ROUND_SPLIT = .8
EARLY_STOPPING_ROUNDS = 20
VERBOSE_INTERVAL = False
NUM_BOOST_ROUND = 1000

class NaiveXGB(Model):
    """
    Basic XGB model.
    """
    def __init__(self, xgb_params=None, log_data=False, name='NaiveXGB', features=None,
                 num_boost_rounds=None):
        self.model = None
        self.num_boost_rounds = num_boost_rounds
        self.log_data = log_data
        if xgb_params is None:
            self.xgb_params = {
                    # 'eta': 0.05,
                    # 'max_depth': 6,
                    # 'subsample': 0.6,
                    # 'colsample_bytree': 1,
                    'objective': 'reg:linear',
                    'eval_metric': 'rmse',
                    'silent': 1
                }
        else:
            self.xgb_params = xgb_params
        self.name = name
        self.features = features

    def find_num_boost_round(self, X_trn, y_trn):
        """
        Find the number of iterations we should run using a mini
        validation set. This way we don't overfit the predictions.
        """
        split = int(NUM_ROUND_SPLIT * X_trn.shape[0])
        d_trn = xgb.DMatrix(X_trn.iloc[:split], label=y_trn.iloc[:split])
        d_val = xgb.DMatrix(X_trn.iloc[split:], label=y_trn.iloc[split:])
        val_model = xgb.train(self.xgb_params, d_trn, 
                  num_boost_round=NUM_BOOST_ROUND,
                  evals=[(d_val, 'val')], 
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                  verbose_eval=VERBOSE_INTERVAL)
        return val_model.best_iteration

    def train(self, X_trn, y_trn):
        if self.features is not None:
            X_trn = X_trn[self.features]
        if self.log_data:
            y_trn = np.log1p(y_trn)
        if self.num_boost_rounds is None:
            self.num_boost_rounds = self.find_num_boost_round(X_trn, y_trn)
        d_trn = xgb.DMatrix(X_trn, y_trn)
        self.model = xgb.train(self.xgb_params, d_trn, 
                               num_boost_round=self.num_boost_rounds)

    def test(self, X_test, y_test=None):
        if self.features is not None:
            X_test = X_test[self.features]
        d_test = xgb.DMatrix(X_test)
        preds = self.model.predict(d_test)
        if self.log_data:
            preds = np.exp(preds) - 1
        return preds