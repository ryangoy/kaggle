import numpy as np
import pandas as pd
from os.path import join
import tensorflow as tf
from utils import load_data, save_output
import time
from model import TSANet

DATADIR     = '/home/ryan/cs/datasets/tsa/stage1'
A3DAPS      = join(DATADIR, 'a3daps')
LABELS      = join(DATADIR, 'formatted_labels.csv')
DATA_H5     = join(DATADIR, 'train_data.h5')
OUTPUT      = join(DATADIR, 'preds.npy')
NUM_CLASSES = 17
NUM_EPOCHS  = 10
BATCH_SIZE  = 4
DISPLAY_STEP= 10
TV_SPLIT    = 0.9

def main():
  # Load data.
  raw_data = load_data(A3DAPS, LABELS, DATA_H5, load_from_h5=True)

  # TODO: Preprocess data.
  data = raw_data

  with tf.Session() as sess:
    # Initialize model.
    tsanet = TSANet(sess, data['data'].shape[1:], NUM_CLASSES)

    # Train and validate.
    tsanet.train_val(data['data'], data['labels'], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                display_step=DISPLAY_STEP, split=TV_SPLIT)

    # Test.
    # preds = tsanet.infer()

  # Format and save.
  #save_output(OUTPUT, preds)

if __name__ == '__main__':
  main()
