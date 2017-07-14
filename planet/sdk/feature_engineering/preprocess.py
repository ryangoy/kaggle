import numpy as np
import pandas as pd
from os.path import join, exists
from os import makedirs
import sys
from shutil import copyfile, rmtree

# TO DO per dataset:
# - change the clean_data method to whatever is needed

def import_clean(ds_dir):
    id_to_labels = one_hot_vectorize_labels(ds_dir)
    generate_binary_classifier_structure(ds_dir)
    return train, test

def one_hot_vectorize_labels(ds_dir):
    """
    For planet competition. Load in training csv and process the
    label column. See one_hot_vectorize_train_csv.ipynb in the ipython folder.
    """
    train = pd.read_csv(join(ds_dir, 'processed/train.csv'))
    return

# UNTESTED
def generate_binary_data_structure(train_folder, target_folder, train_df, feature_name,
                                   extension='.jpg'):
    """
    Copies data from train_folder to another folder where it puts images into two subfolders:
    one for positive images for feature_name and one for negatives.
    """
    # delete existing folders
    if exists(target_folder):
        rmtree(target_folder)
    os.makedirs(target_folder)
    negatives_dir = join(target_folder, 'not_'+feature_name)
    positives_dir = join(target_folder, feature_name)
    os.makedirs(negatives_dir)
    os.makedirs(positives_dir)
  
    # sort each picture into its respective directory
    for row in train_df:
        if row[feature_name] == 1:
            dest_dir = positives_dir
        else:
            dest_dir = negatives_dir
        copyfile(join(train_folder, row.image_name+extension), 
                 join(target_folder, dest_dir))
        
def import_data():
    train = pd.read_csv(join(ds_dir, 'train.csv'))
    test = pd.read_csv(join(ds_dir, 'test.csv'))
    return train, test

def clean_mercedes_data(train, test):
    useless_columns = ['X11', 'X93', 'X107', 'X233', 'X235', 
                       'X268', 'X289', 'X290', 'X293', 'X297', 
                       'X330', 'X347']
    train = train.drop(useless_columns, axis=1)
    test = test.drop(useless_columns, axis=1)
    #train = train[train.y <= 200]
    train[train.y >= 200] = 125

    return train, test
