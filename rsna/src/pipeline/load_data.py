import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import io
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
import keras
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def load_trn_data(images_path, labels_path, val_split=0.1):

    # empty dictionary
    pneumonia_locations = {}
    # load table
    with open(labels_path, mode='r') as infile:
        # open reader
        reader = csv.reader(infile)
        # skip header
        next(reader, None)
        # loop through rows
        for rows in reader:
            # retrieve information
            filename = rows[0]
            location = rows[1:5]
            pneumonia = rows[5]
            # if row contains pneumonia add label to dictionary
            # which contains a list of pneumonia locations per filename
            if pneumonia == '1':
                # convert string to float to int
                location = [int(float(i)) for i in location]
                # save pneumonia location in dictionary
                if filename in pneumonia_locations:
                    pneumonia_locations[filename].append(location)
                else:
                    pneumonia_locations[filename] = [location]


    # load and shuffle filenames
    folder = images_path
    filenames = os.listdir(folder)
    random.shuffle(filenames)

    split_index = int((1-val_split)*len(filenames))

    trn_filenames = filenames[:split_index]
    val_filenames = filenames[split_index:]


    trn_gen = RSNAGenerator(folder, trn_filenames, pneumonia_locations, batch_size=32, image_size=256, shuffle=True, augment=True, predict=False)
    val_gen = RSNAGenerator(folder, val_filenames, pneumonia_locations, batch_size=32, image_size=256, shuffle=False, predict=False)

    return trn_gen, val_gen

def load_test_data(path='/home/ryan/cs/datasets/rsna/stage_1_test_images'):
    # load and shuffle filenames
    
    filenames = os.listdir(path)

    return filenames


def load_trn_metadata():
    labels = pd.read_csv('/home/ryan/cs/datasets/rsna/stage_1_train_labels.csv')
    details = pd.read_csv('/home/ryan/cs/datasets/rsna/stage_1_detailed_class_info.csv')
    # duplicates in details just have the same class so can be safely dropped
    details = details.drop_duplicates('patientId').reset_index(drop=True)
    labels_w_class = labels.merge(details, how='inner', on='patientId')

    train_dcm_fps = glob.glob('/home/ryan/cs/datasets/rsna/stage_1_train_images/*.dcm')
    train_dcms = [pydicom.read_file(x, stop_before_pixels=True) for x in train_dcm_fps]
    train_meta_dicts, tag_to_keyword_train = zip(*[parse_dcm_metadata(x) for x in train_dcms])
    unified_tag_to_key_train = {k:v for dict_ in tag_to_keyword_train for k,v in dict_.items()}
    train_df = pd.DataFrame.from_records(data=train_meta_dicts)

    train_df = clean_metadata(train_df)
    return train_df

def load_test_metadata():
    test_dcm_fps = glob.glob('/home/ryan/cs/datasets/rsna/stage_1_test_images/*.dcm')
    test_dcms = [pydicom.read_file(x, stop_before_pixels=True) for x in test_dcm_fps]
    test_meta_dicts, tag_to_keyword_test = zip(*[parse_dcm_metadata(x) for x in test_dcms])
    unified_tag_to_key_test = {k:v for dict_ in tag_to_keyword_test for k,v in dict_.items()}
    test_df = pd.DataFrame.from_records(data=test_meta_dicts)
    test_df['PixelSpacing_x'] = df['PixelSpacing'].apply(lambda x: x[0])
    test_df['PixelSpacing_y'] = df['PixelSpacing'].apply(lambda x: x[1])
    test_df = df.drop(['PixelSpacing'], axis='columns')
    assert sum(train_df['ReferringPhysicianName'] != '') == 0
    set(test_df['SeriesDescription'].unique())

    test_df = clean_metadata(test_df)
    return test_df

def clean_metadata(df):

    df['PixelSpacing_x'] = df['PixelSpacing'].apply(lambda x: x[0])
    df['PixelSpacing_y'] = df['PixelSpacing'].apply(lambda x: x[1])
    df = df.drop(['PixelSpacing'], axis='columns')
    assert sum(df['ReferringPhysicianName'] != '') == 0
    # drop constant cols and other two from above
    df = df.drop(nunique_all[nunique_all == 1].index.tolist() + ['ReferringPhysicianName', 'SeriesDescription'], axis='columns')

    # now that we have a clean metadata dataframe we can merge back to our initial tabular data with target and class info
    df = df.merge(labels_w_class, how='left', left_on='PatientID', right_on='patientId')

    df['PatientAge'] = df['PatientAge'].astype(int)

    # df now has multiple rows for some patients (those with multiple bounding boxes in label_w_class)
    # so creating one with no duplicates for patients
    df_deduped = df.drop_duplicates('PatientID', keep='first')

    return df

def parse_dcm_metadata(dcm):
    unpacked_data = {}
    group_elem_to_keywords = {}
    # iterating here to force conversion from lazy RawDataElement to DataElement
    for d in dcm:
        pass
    # keys are pydicom.tag.BaseTag, values are pydicom.dataelem.DataElement
    for tag, elem in dcm.items():
        tag_group = tag.group
        tag_elem = tag.elem
        keyword = elem.keyword
        group_elem_to_keywords[(tag_group, tag_elem)] = keyword
        value = elem.value
        unpacked_data[keyword] = value
    return unpacked_data, group_elem_to_keywords

class RSNAGenerator(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=256, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in self.pneumonia_locations:
            # loop through pneumonia
            for location in self.pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)