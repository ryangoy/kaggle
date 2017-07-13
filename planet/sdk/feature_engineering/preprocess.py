import numpy as np
import pandas as pd
from os.path import join
import sys

# TO DO per dataset:
# - change the clean_data method to whatever is needed

def import_clean(ds_dir):
    id_to_labels = one_hot_vectorize_labels(ds_dir)
    generate_binary_classifier_structure(ds_dir)
    return train, test

def one_hot_vectorize_labels(ds_dir):
    """
    For planet competition. Load in training csv and process the
    label column.
    """
    labels_file = join(ds_dir, 'orig/train_v2.csv')
    return

def generate_binary_classifier_structure(ds_dir):
    """
    Copies data from original data to another folder where each subfolder
    is the label.
    """
    orig_folder = join(ds_dir, 'orig/train-tif-v2')
    target_folder = join(ds_dir, 'binary')
    return

def import_data():
    train = pd.read_csv(join(ds_dir, 'train.csv'))
    test = pd.read_csv(join(ds_dir, 'test.csv'))
    return train, test

def clean_data(train, test):
    useless_columns = ['X11', 'X93', 'X107', 'X233', 'X235', 
                       'X268', 'X289', 'X290', 'X293', 'X297', 
                       'X330', 'X347']
    train = train.drop(useless_columns, axis=1)
    test = test.drop(useless_columns, axis=1)
    #train = train[train.y <= 200]
    train[train.y >= 200] = 125

    return train, test

