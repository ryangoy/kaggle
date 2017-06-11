import numpy as np
import pandas as pd
from feature_engineering import encode_categorical_features, \
    generate_time_features,\
    generate_relative_square_footage,\
    generate_room_information
from preprocess import import_clean
from utils import KFold




def run():
    # feature engineering
    train, test, macro = import_clean()
    encode_categorical_features(train, test)
    generate_time_features(train, test)
    generate_relative_square_footage(train, test)
    generate_room_information(train, test)

    # drop unnecessary features

    # train
    xgb_model = NaiveXGB()
    kf = KFold(train.drop(['price_doc', 'id']), train['price_doc'], num_folds=2)
    preds = kf.run_kfolds_on_model(xgb_model)
    print preds.shape

    # test

# boilerplate code
if __name__ == '__main__':
    run()
