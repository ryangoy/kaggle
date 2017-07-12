import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.svm import LinearSVC
SEED = 0

class KFold:

    def __init__(self, X_trn, y_trn, X_test, num_folds=5):
        self.X_trn = X_trn
        self.y_trn = y_trn
        self.X_test = X_test
        kf = model_selection.KFold(n_splits=num_folds, random_state=SEED, shuffle=True)
        self.splits = list(kf.split(X_trn))
        #self.feature_select()

    def feature_select(self):
        #sel = SelectPercentile(f_regre)
        sel = SelectKBest(f_regression, k=100)
        self.X_trn = pd.DataFrame(sel.fit_transform(self.X_trn, self.y_trn))
        self.X_test = pd.DataFrame(sel.transform(self.X_test))

    def run_kfolds_on_model(self, model):
        pred_indices = np.array([], dtype=int)
        all_val_preds = np.array([])
        test_preds = pd.DataFrame()

        # loop through folds
        index = 0
        for trn_indices, val_indices in self.splits:
            # set the train and test sets
            X_trn = self.X_trn.iloc[trn_indices]
            y_trn = self.y_trn.iloc[trn_indices]
            X_val = self.X_trn.iloc[val_indices]
            y_val = self.y_trn.iloc[val_indices]

            # train the model for this fold
            model.train(X_trn, y_trn, X_val, y_val)
            val_preds = model.test(X_val, y_val)
            all_val_preds = np.append(all_val_preds, val_preds)
            pred_indices = np.append(pred_indices, val_indices)
            test_preds[str(index)] = model.test(self.X_test)
            index += 1

        # un-shuffle the predictions
        sorted_indices = np.argsort(pred_indices)
        return all_val_preds[sorted_indices], test_preds.mean(axis=1) # maybe use median instead?



