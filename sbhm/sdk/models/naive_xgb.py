import numpy as np
import pandas as pd
import xgboost as xgb
import Model

NUM_ROUND_SPLIT = .8
EARLY_STOPPING_ROUNDS = 20
VERBOSE_INTERVAL = 25
NUM_BOOST_ROUND = 1000

class NaiveXGB(Model):
    """
    Basic XGB model.
    """
    def __init__(self):
        self.model = None
        self.num_boost_rounds = None
        self.log_data = True
        self.xgb_params = {
                'eta': 0.05,
                'max_depth': 5,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'objective': 'reg:linear',
                'eval_metric': 'rmse',
                'silent': 0
            }

    def __init__(self, xgb_params, log_data=True, name='NaiveXGB'):
        self.model = None
        self.xgb_params = xgb_params
        self.log_data = log_data
        self.name = name

    def find_num_boost_round(X_trn, y_trn):
        """
        Find the number of iterations we should run using a mini
        validation set. This way we don't overfit the predictions.
        """
        split = int(NUM_ROUND_SPLIT * X_trn.shape[0])
        d_trn = xgb.DMatrix(X_trn[:split], y_trn[:split])
        d_val = xgb.DMatrix(X_trn[split], y_trn[split:])
        val_model = xgb.train(self.xgb_params, d_trn, 
                  num_boost_round=NUM_BOOST_ROUND,
                  evals=[(d_val, 'val')], 
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                  verbose_eval=VERBOSE_INTERVAL)
        return val_model.best_iteration

    def train(self, X_trn, y_trn):
        if log_data:
            y_trn = np.log1p(y_trn)
        if self.num_boost_rounds is None:
            self.num_boost_rounds = find_num_boost_round(X_trn, y_trn)
        d_trn = xgb.DMatrix(X_trn, y_trn)
        self.model = xgb.train(xgb_params, d_trn, 
                               num_boost_round=self.num_boost_rounds)

    def test(self, X_test, y_test=None):
        d_test = xgb.DMatrix(X_test)
        preds = self.model.predict(d_test)
        if self.log_data:
            preds = np.exp(preds) - 1
        return preds