import sys
sys.path.insert(0, '../feature_generation')
sys.path.insert(0, '../..')
import pandas as pd
import numpy as np
import keras
import models
import paths
import shutil
from PIL import Image
import math
from os.path import join
from os import listdir
from keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection
import matplotlib.pyplot as plt
import train_cnn
import run_val_cnn

df = pd.read_json(paths.TRAIN_JSON)
listing_id_interest = df[['listing_id', 'interest_level']]

trn_indices = []
val_indices = []

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
categories = ['low', 'medium', 'high']
count = 0
for dev_index, val_index in kf.split(range(len(listing_id_interest))):
    # if count !=3:
    #     count += 1
    #     continue
    print "Beginning split {}".format(count)
    trn_indices = dev_index
    val_indices = val_index
    ids = listing_id_interest['listing_id']
    ids = np.array(ids)
    val_ids = ids[val_indices]
    # move everything back into train
    for folder in categories:
        for img in listdir(join(paths.VAL_IMAGES, folder)):
            shutil.move(join(paths.VAL_IMAGES,folder,img), 
                        join(paths.TRN_IMAGES,folder,img))
    # creating val set
    for folder in categories:
        for img in listdir(join(paths.TRN_IMAGES, folder)):
            listing_id = int(img[:img.index('_')])
            if listing_id in val_ids:
                shutil.move(join(paths.TRN_IMAGES,folder,img), 
                            join(paths.VAL_IMAGES,folder,img))
    print "Organized dataset successfully. Now training..."
    train_cnn.train(save_path = join(paths.CNN_OUTPUT_DIR, 
                                     "kfold_{}.h5".format(count)))
    run_val_cnn.test(save_id_path=join(paths.CNN_OUTPUT_DIR,
                                     "kfold_{}_ids.npy".format(count)),
                   save_path = join(paths.CNN_OUTPUT_DIR,
                                     "kfold_{}_preds.npy".format(count)))
    print "Testing..."
    count += 1
"Finished!"
