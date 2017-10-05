import numpy as np
import pandas as pd
from os.path import join
import tensorflow as tf
from utils import load_data, save_output
import time

DATADIR     = '/home/ryan/cs/datasets/tsa/stage1'
A3DAPS      = join(DATADIR, 'a3daps')
LABELS      = join(DATADIR, 'formatted_labels.csv')
DATA_H5     = join(DATADIR, 'data.h5')
NUM_CLASSES = 17
NUM_EPOCHS  = 10
BATCH_SIZE  = 4


def main():
  # Load data.
  raw_data = load_data(A3DAPS, DATA_H5, load_from_npy=True)

  # Initialize model.
  tsanet = TSANet(data[0].shape[1:], NUM_CLASSES)

  # Train and validate.
  tsanet.train_val()

  # Test.
  preds = tsanet.infer()

  # Format and save.
  save_output(preds)

if __name__ == '__main__':
  main()
