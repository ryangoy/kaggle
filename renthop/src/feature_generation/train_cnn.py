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

model = models.VGG16_train(classes = 3)

datagen = ImageDataGenerator(
    horizontal_flip=True)

trn_gen = datagen.flow_from_directory(paths.IMAGES, 
                                    classes=['low','medium','high'],
                                    seed =0, target_size=(224, 224))
val_gen = datagen.flow_from_directory(join(paths.IMAGES,'val'), 
                                    classes=['low','medium','high'],
                                    seed =0, target_size=(224, 224))

model.fit_generator(trn_gen, samples_per_epoch = 221317 / 32, 
                    validation_data = val_gen,
                    nb_val_samples = 54914 / 32,
                    nb_epoch = 10, verbose=1,
                    callbacks=[], nb_worker=1)


model.save(paths.VGG16_RENTHOP_WEIGHTS)
