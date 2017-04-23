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
from keras.models import load_model
from sklearn import model_selection
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(save_path = paths.PREDICTIONS_DIR):
    datagen = ImageDataGenerator()
    model = load_model(paths.VGG16_RENTHOP_WEIGHTS)
    test_gen = datagen.flow_from_directory(paths.IMGS_CATEGORIZED, 
                                        classes=['test'],
                                        seed =0, target_size=(224, 224),
                                        shuffle=False, batch_size=32)
    preds = model.predict_generator(test_gen, 418591/32 + 1, verbose=1)
    np.save(join(save_path, 'cnn_test_filenames.npy'), test_gen.filenames)
    np.save(join(save_path, 'cnn_test_preds.npy'), preds)

if __name__ == '__main__':
    test()
