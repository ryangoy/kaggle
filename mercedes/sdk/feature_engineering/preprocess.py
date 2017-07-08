import numpy as np
import pandas as pd
from os.path import join
import sys

# TO DO per dataset:
# - change the clean_data method to whatever is needed

def import_clean(ds_dir):
    train = pd.read_csv(join(ds_dir, 'train.csv'))
    test = pd.read_csv(join(ds_dir, 'test.csv'))
    train, test = clean_data(train, test)

    return train, test


def clean_data(train, test):
    useless_columns = ['X11', 'X93', 'X107', 'X233', 'X235', 
                       'X268', 'X289', 'X290', 'X293', 'X297', 
                       'X330', 'X347']
    train = train.drop(useless_columns, axis=1)
    test = test.drop(useless_columns, axis=1)
    train.y[train.y > 200] = 200

    return train, test

