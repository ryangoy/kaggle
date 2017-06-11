import numpy as np
import pandas as pd
from sklearn import model_selection

SEED = 0

class KFold:

    def __init__(self, X, y, num_folds=5):
        self.X = X
        self.y = y
        kf = model_selection.KFold(n_splits=num_folds, random_state=SEED, shuffle=True)
        self.trn_splits, self.test_splits = kf.split(X)

    def run_kfolds_on_model(self, model):
        pred_indices = np.array([], dtype=int)
        all_preds = np.array([])

        # loop through folds
        for trn_indices, test_indices in zip(self.trn_splits, self.test_splits):
            # set the train and test sets
            X_trn = self.X.iloc[trn_indices]
            y_trn = self.y.iloc[trn_indices]
            X_test = self.X.iloc[test_indices]
            y_test = self.y.iloc[test_indices]

            # train the model for this fold
            model.train(X_trn, y_trn)
            preds = model.test(X_test, y_test)
            all_preds = np.append(all_preds, preds)
            pred_indices = np.append(pred_indices, test_indices)
            print test_indices[:10]

        # un-shuffle the predictions
        return all_preds[pred_indices]



