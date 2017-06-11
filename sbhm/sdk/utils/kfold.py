import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 0

class KFold:

    def __init__(self, X, y, num_folds=5):
        self.X = X
        self.y = y
        kf = KFold(n_splits=num_folds, random_state=SEED, shuffle=True)
        trn_splits, test_splits = kf.split(X_trn)

    def run_kfolds_on_model(self, model):
        pred_indices = []
        all_preds = []

        # loop through folds
        for trn_indices, test_indices in zip(trn_splits, test_splits):
            # set the train and test sets
            X_trn = self.X[trn_indices]
            y_trn = self.y[trn_indices]
            X_test = self.X[test_indices]
            y_test = self.y[test_indices]

            # train the model for this fold
            model.train(X_trn, y_trn)
            preds = model.test(X_test, y_test)
            all_preds += preds
            pred_indices += test_indices

        # un-shuffle the predictions
        return all_preds[pred_indices]



