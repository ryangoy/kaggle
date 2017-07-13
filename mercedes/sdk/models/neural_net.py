import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from model import Model
from sklearn import preprocessing

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
        self.scaler = None

    def init_model(self, num_cols):
        # create model
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=num_cols, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        self.model.compile(loss='mse', optimizer='adam', lr=0.001)

    def train(self, X_trn, y_trn, X_val=None, y_val=None):
        self.scaler = preprocessing.MinMaxScaler()
        x_scaled = self.scaler.fit_transform(X_trn.values)
        X_trn = pd.DataFrame(x_scaled)
        X_trn = X_trn.fillna(X_trn.median())
        if X_val is not None and y_val is not None:
            X_val = self.scaler.transform(X_val.values)
        if self.features is not None:
            X_trn = X_trn[self.features]
            if X_val is not None:
                X_val = X_val[self.features]
        if self.log_data:
            y_trn = np.log1p(y_trn)
        self.init_model(len(X_trn.columns))
        self.model.fit(X_trn.as_matrix(), y_trn.as_matrix(), epochs=self.epochs, 
            batch_size=self.batch_size, validation_data=(X_val, y_val), verbose=1)
        
    def test(self, X_test, y_test=None):
        X_test = X_test.fillna(X_test.median()).fillna(0)
        X_test = pd.DataFrame(self.scaler.transform(X_test.values))
        if self.features is not None:
            X_test = X_test[self.features]
        preds = self.model.predict(X_test.as_matrix(), batch_size=self.batch_size)
        preds = preds.reshape((-1,))
        if self.log_data:
            preds = np.exp(preds) - 1
        return preds