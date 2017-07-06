import numpy as np
import pandas as pd
import xgboost as xgb
from model import Model

NUM_ROUND_SPLIT = .8
EARLY_STOPPING_ROUNDS = 50
VERBOSE_INTERVAL = False
NUM_BOOST_ROUND = 2000

# note: not using find_num_boost_round feature currently

class NaiveXGB(Model):
    """
    Basic XGB model.
    """
    def __init__(self, xgb_params=None, log_data=False, name='NaiveXGB', features=None,
                 num_boost_rounds=NUM_BOOST_ROUND, custom_eval=None, maximize=False):
        self.model = None
        self.num_boost_rounds = num_boost_rounds
        self.log_data = log_data
        if xgb_params is None:
            self.xgb_params = {
                    'objective': 'reg:linear',
                    'silent': 1
                }
        else:
            self.xgb_params = xgb_params
        self.name = name
        self.features = features
        self.custom_eval = custom_eval
        self.maximize = maximize

    def generate_feval(self, loss_fn, name='custom_loss'):
        if loss_fn is None:
            return None
        def xgb_loss_function(preds, d_trn):
            labels = d_trn.get_label()
            return name, loss_fn(labels, preds)
        return xgb_loss_function

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

    def train(self, X_trn, y_trn, X_val=None, y_val=None):
        if self.features is not None:
            X_trn = X_trn[self.features]
        if self.log_data:
            y_trn = np.log1p(y_trn)
        if self.num_boost_rounds is None:
            self.num_boost_rounds = self.find_num_boost_round(X_trn, y_trn)
        xgb_loss_function = None
        if self.custom_eval is not None:
            xgb_loss_function = self.generate_feval(self.custom_eval)
        d_trn = xgb.DMatrix(X_trn, y_trn)
        if X_val is None or y_val is None:
            watchlist = [(d_trn, 'train')]
        else:
            d_val = xgb.DMatrix(X_val, y_val)
            watchlist = [(d_trn, 'train'), (d_val, 'valid')]
        self.model = xgb.train(self.xgb_params, d_trn, 
                               num_boost_round=self.num_boost_rounds,
                               evals = watchlist,
                               feval=xgb_loss_function,
                               early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                               maximize=self.maximize,
                               verbose_eval=100)

    def test(self, X_test, y_test=None):
        if self.features is not None:
            X_test = X_test[self.features]
        d_test = xgb.DMatrix(X_test)
        preds = self.model.predict(d_test)
        if self.log_data:
            preds = np.exp(preds) - 1
        return preds