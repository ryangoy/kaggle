import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from model import Model

np.random.seed(0)

class NeuralNet(Model):
    """
    Basic neural net model.
    """
    def __init__(self, log_data=False, name='NeuralNet', features=None, epochs=10, 
        batch_size=16):
        self.model = None
        self.log_data = log_data
        self.name = name
        self.features = features
        self.epochs = epochs
        self.batch_size = batch_size

    def init_model(self, num_cols):
        # create model
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=num_cols, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X_trn, y_trn, X_val=None, y_val=None):
        X_trn = X_trn.fillna(X_trn.median())
        if self.features is not None:
            X_trn = X_trn[self.features]
        if self.log_data:
            y_trn = np.log1p(y_trn)
        self.init_model(len(X_trn.columns))
        self.model.fit(X_trn.as_matrix(), y_trn.as_matrix(), epochs=self.epochs, batch_size=self.batch_size)

    def test(self, X_test, y_test=None):
        X_test = X_test.fillna(X_test.median()).fillna(0)

        if self.features is not None:
            X_test = X_test[self.features]
        preds = self.model.predict(X_test.as_matrix(), batch_size=16)
        preds = preds.reshape((-1,))
        if self.log_data:
            preds = np.exp(preds) - 1
        return pd.Series(preds)