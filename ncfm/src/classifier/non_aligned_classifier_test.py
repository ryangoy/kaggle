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

test_dir = join(dataset_dir, 'test_stg2')
weights_path = join(dataset_dir, 'classifier_weights.h5')
final_save_path = join(dataset_dir, 'blended/FINAL_PREDS.npy')
confidences = join(dataset_dir, 'blended/FINAL_CONF.npy')
#loaded_images = join(dataset_dir, 'loaded_test_images.npy')

seed = 0
np.random.seed(seed)
vgg_size = (224, 224)
fish_types = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
nb_classes = len(fish_types)
test_bboxes = '/home/ryan/cs/datasets/ncfm/blended/test_boxes.pkl'


def load_images():
    results = pickle.load(open(test_bboxes, 'r'))
    images = []
    for fname in sorted(os.listdir(test_dir)):
        if fname == ".DS_Store":
            continue
        img_path = os.path.join(test_dir, fname)
        img = image.load_img(img_path, target_size=vgg_size)
        img = image.img_to_array(img)
        images.append(imread(img_path))
    #images = np.load(loaded_images)
    #np.save(loaded_images, images)
    #print "SAVED IMAGES!"
    # print images.shape
    # print len(results)
    cropped_images = []
    fish_confidence = []
    counter = 0 
    for i, img in enumerate(images):
        
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]
        
        k = det_conf.argsort()[::-1]
        top_indices = k[:1]
        
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            
            cropped_images.append(get_cropped_fish(img, xmin, ymin, xmax, ymax))
            fish_confidence.append(top_conf[i])

    test_images = np.asarray(cropped_images)

    return test_images, fish_confidence

def get_cropped_fish(img,x1,y1,x2,y2):
    (h,w) = img.shape[:2]

    center = ( (x1+x2) / 2,(y1+y2) / 2)
    
    fish_length = np.sqrt((x1-x2)**2+(y1-y2)**2)
    
    cropped = img[(max((center[1]-int(fish_length/1.8)),0)):(max((center[1]+int(fish_length/1.8)),0)) ,
                      (max((center[0]-int(fish_length/1.8)),0)):(max((center[0]+int(fish_length/1.8)),0))]
    resized = resize(cropped,vgg_size)
    return resized


images, fish_confidence = load_images()

images *= 255
images = images[:, :, :, ::-1]

model = models.VGG16_test(weights_path, input_shape=vgg_size+ (3,), classes=nb_classes,)

final_preds = model.predict(images)

np.save(final_save_path, final_preds)
np.save(confidences, fish_confidence)

