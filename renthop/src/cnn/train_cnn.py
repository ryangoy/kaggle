import sys
sys.path.insert(0, '../..')
import pandas as pd
import numpy as np
import keras
import models
import paths
from PIL import Image
import math
from os.path import join
from os import listdir
from keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection
import matplotlib.pyplot as plt
from PIL import ImageFile
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(save_path = paths.VGG16_RENTHOP_WEIGHTS, use_val=True):
    model = models.VGG16_train(classes = 3)

    datagen = ImageDataGenerator(
        horizontal_flip=True)

    trn_gen = datagen.flow_from_directory(paths.TRN_IMAGES, 
                                        classes=['low','medium','high'],
                                        seed =0, target_size=(224, 224))
    if use_val:
        val_gen = datagen.flow_from_directory(paths.VAL_IMAGES, 
                                            classes=['low','medium','high'],
                                            seed =0, target_size=(224, 224),
                                            batch_size=32)


        model.fit_generator(trn_gen, samples_per_epoch = 221477 / 32, 
                            validation_data = val_gen,
                            nb_val_samples = 100,
                            nb_epoch = 6, verbose=1,
                            callbacks=[], nb_worker=1)
    else:
        model.fit_generator(trn_gen, samples_per_epoch = 276431 / 32 + 1, 
                            nb_epoch = 6, verbose=1,
                            callbacks=[], nb_worker=1)


    model.save(save_path)

def train_on_entire_data():
    # move everything into train
    categories = ['low','medium','high']
    for folder in categories:
        for img in listdir(join(paths.VAL_IMAGES, folder)):
            shutil.move(join(paths.VAL_IMAGES,folder,img), 
                        join(paths.TRN_IMAGES,folder,img))
    train(use_val=False)

if __name__ == '__main__':
    train_on_entire_data()
