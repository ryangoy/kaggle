import numpy as np
import os
from scipy.misc import imread
import cv2

def load_tgs_data(path, train=True):
	images_path = os.path.join(path, 'images')
	images_names = os.listdir(images_path)

	images = []
	for i in images_names:
		raw = imread(os.path.join(images_path, i), mode='L')
		raw = cv2.resize(raw, (128, 128))
		images.append(raw.reshape(128, 128, 1))

	masks = None
	if train:
		masks = []
		masks_path = os.path.join(path, 'masks')
		masks_names = os.listdir(masks_path)
		for m in masks_names:
			raw = imread(os.path.join(masks_path, m), mode='L')
			raw = cv2.resize(raw, (128, 128))
			masks.append(raw.reshape(128, 128, 1))

	return np.array(images), np.array(masks)

	
	

