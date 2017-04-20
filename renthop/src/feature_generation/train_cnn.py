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
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(save_path = paths.VGG16_RENTHOP_WEIGHTS):
    model = models.VGG16_train(classes = 3)

    datagen = ImageDataGenerator(
        horizontal_flip=True)

    trn_gen = datagen.flow_from_directory(paths.TRN_IMAGES, 
                                        classes=['low','medium','high'],
                                        seed =0, target_size=(224, 224))
    val_gen = datagen.flow_from_directory(paths.VAL_IMAGES, 
                                        classes=['low','medium','high'],
                                        seed =0, target_size=(224, 224),
                                        batch_size=32)


    model.fit_generator(trn_gen, samples_per_epoch = 221477 / 32, 
                        validation_data = val_gen,
                        nb_val_samples = 100,
                        nb_epoch = 6, verbose=1,
                        callbacks=[], nb_worker=1)


    model.save(save_path)
