import numpy as np
import pandas as pd

class KFold:
  def __init__(self, num_folds, X_images, X_metadata, y):
    self.num_folds = num_folds
    self.set_size = len(X_images)
    assert self.set_size == len(X_metadata)
    assert self.set_size == len(y)
    self.create_folds()

  def create_folds(self):

    self.split_indices = [i*int(self.set_size/self.num_folds) for i in range(self.num_folds)]
    self.image_folds = [X_images[self.split_indices[i]:self.split_indices[i+1]]for i in range(num_folds-1)]
    self.image_folds.append([X_images[self.split_indices[-1]:]])

    self.metadata_folds = [X_metadata[self.split_indices[i]:self.split_indices[i+1]]for i in range(num_folds-1)]
    self.metadata_folds.append([X_metadata[self.split_indices[-1]:]])

    self.label_folds = [y[self.split_indices[i]:self.split_indices[i+1]]for i in range(num_folds-1)]
    self.label_folds.append([y[self.split_indices[-1]:]])

    self.curr_fold = 0


  def get_trn_fold(self, fold_id):
    images = [i for i in self.image_folds if i != fold_id]
    metadata = [i for i in self.metadata_folds if i != fold_id]
    labels = [i for i in self.label_folds if i != fold_id]
    return images, metadata, labels
    

  def get_val_fold(self, fold_id):
    return self.image_folds[fold_id], self.metadata_folds[fold_id], self.label_folds[fold_id]


