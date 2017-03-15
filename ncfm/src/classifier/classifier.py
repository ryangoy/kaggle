import cv2
import numpy as np
import os
import sklearn
import time
from keras.callbacks import ModelCheckpoint
import sys
from skimage.data import imread
from skimage.io import imshow,imsave
from skimage import img_as_float
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from os.path import join
from skimage.util import crop
from skimage.transform import rotate
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
import pickle
from math import atan2, degrees, pi
from keras.preprocessing import image
from keras import backend as K

sys.path.append('..')
from lib import models
from lib import utils

dataset_dir = '/home/ryan/cs/datasets/ncfm'

label_dir = '/home/ryan/cs/kaggle/ncfm/point_annotations'

img_dir = join(dataset_dir, 'train')

weights_path = join(dataset_dir, 'weights/vgg16_full.h5')

images_path = join(dataset_dir, 'classifier_images.npy')
labels_path = join(dataset_dir, 'classifier_labels.npy')

load_precomputed_train = True
seed = 0
np.random.seed(seed)
vgg_size = (224, 224)
fish_types = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
nb_test_samples = 1000
nb_classes = len(fish_types)
k = 3
nb_epoch_e2e = 15
nb_epoch_localized = 15
nb_epoch = 15
nb_runs = 5
batch_size = 16


json_names = ['alb_labels.json',
              'bet_labels.json',
              'dol_labels.json',
              'lag_labels.json',
              'other_labels.json',
              'shark_labels.json',
              'yft_labels.json',]

label_files = []
for n in json_names:
    label_files.append(join(label_dir, n))


def get_angle(row):
    if len(row['annotations']) < 2:
        return float('NaN')
    p1 = row['annotations'][0]
    p2 = row['annotations'][1]
    return deg_angle_between(p1['x'],p1['y'], p2['x'], p2['y'])

def deg_angle_between(x1,y1,x2,y2):
    
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
#     if degs > 180:
#         degs -= 360
    return(degs)


def get_img(row):
    if len(row['annotations']) < 2:
        return float('NaN')
    p1 = row['annotations'][0]
    p2 = row['annotations'][1]

    img = imread(join(img_dir,row['filename']))
    #img = image.load_img(join(img_dir,row['filename']), target_size=(224, 224))
    #x = image.img_to_array(img)
    cropped = get_cropped_fish(img, 
        int(p1['x']), int(p1['y']), int(p2['x']), int(p2['y']))
    #x = np.expand_dims(x, axis=0)
    # plt.imshow(cropped)
    # plt.show()
    return cropped
    


def get_cropped_fish(img,x1,y1,x2,y2):
    (h,w) = img.shape[:2]
    #calculate center and angle
    center = ( (x1+x2) / 2,(y1+y2) / 2)
    angle = np.floor(-deg_angle_between(x1,y1,x2,y2))
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
#     plt.imshow(rotated)
#     plt.show()
    fish_length = np.sqrt((x1-x2)**2+(y1-y2)**2)
    
    cropped = rotated[(max((center[1]-int(fish_length/1.8)),0)):(max((center[1]+int(fish_length/1.8)),0)) ,
                      (max((center[0]-int(fish_length/1.8)),0)):(max((center[0]+int(fish_length/1.8)),0))]
    resized = resize(cropped,vgg_size)
    
    return resized

if not load_precomputed_train:
    labels = None
    for i in range(len(label_files)):
        f_labels = pd.read_json(label_files[i])
        f_labels['fish'] = i
        if labels is None:
            labels = f_labels
        else:
            labels = labels.append(f_labels)

    labels['rotation'] = labels.apply(get_angle, axis=1)
    labels = labels.dropna()

    print 'Calculated rotations.'

    images = np.zeros((len(labels), vgg_size[0], vgg_size[1], 3))
    counter = 0
    for _, row in labels.iterrows():
        images[counter] = get_img(row)
        if counter % 1000 == 0:
            print 'Finished processing {} images'.format(counter)
        counter += 1
    #images = preprocess_input(images)
    images -= images.mean(axis=(1,2),keepdims=1)
    
    enc = OneHotEncoder(sparse=False)
    one_hot = enc.fit_transform(labels['fish'].values.reshape(-1,1))
    data = [images, one_hot]
    np.save(open(images_path, 'w'), images)
    np.save(open(labels_path, 'w'), one_hot)
    #pickle.dump(data, open(aligned_images_pkl, 'w'))
else:
    print "Loading precomputed data..."
    #data = pickle.load(open(aligned_images_pkl, 'r'))
    images = np.load(images_path)
    one_hot = np.load(labels_path)
    print "Loaded precomputed data successfully."

#images = images.transpose((0,3,1,2))
images *= 255
images = images[:, :, :, ::-1]

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(images, one_hot, test_size=.2)

# for i in range(10):
#     gg = X_train[i]
#     ll = y_train[i]
#     print gg.shape
#     print ll.shape
#     plt.imshow(gg)
#     plt.show()
#     print ll
trn_gen, val_gen = models.get_train_val_gens(X_train, X_val, y_train, y_val, 
                                             size=vgg_size, batch_size = batch_size)
# k = trn_gen.next()
# test_batch = k[0]
# test_labels = k[1]
# print len(test_batch)
# for i in range(10):
#     gg = test_batch[i]
#     ll = test_labels[i]
#     print gg.shape
#     print ll.shape
#     plt.imshow(gg)
#     plt.show()
#     print ll

model = models.VGG16(weights_path, input_shape=vgg_size+ (3,), classes=7,)

models.train_val(model, trn_gen, val_gen, 
          nb_trn_samples=X_train.shape[0],
          nb_val_samples=X_val.shape[0],
          nb_epoch=20)
# model.fit(X_train, y_train,
#         batch_size=batch_size,
#         nb_epoch=nb_epoch,
#         validation_data=(X_val, y_val),
#         shuffle=True, verbose=1)