import numpy as np
import pandas as pd
import sys
import os.path
import skimage.io

def load_processed_fish(pathname, target_size):
    num_images = 3000
    label_bounds = [0, 199, 1910, 2641, 2758, 2933, 3000] #not using other fish
    label_counts = [label_bounds[i+1]-label_bounds[i] for i in range(len(label_bounds)-1)]
    
    fish_images = []
    for label in range(len(label_counts)):
        images = np.empty((label_counts[label], 224, 224, 3))
        for i in range(label_counts[label]):
            images[i,:,:,:] = skimage.io.imread(os.path.join(pathname, 
                "img_{0}label_{1}.jpg".format(i + label_bounds[label], label)))
        resize_image_set(images, target_size)
        fish_images.append(images)
    X = np.array(fish_images).reshape((-1,224,224,3))
    y = []
    for i in range(len(label_counts)):
        y += [i] * target_size
    y = np.array(y)

    print "Loaded training images with shape {0}.".format(X.shape)
    print "Loaded training labels with shape {0}.".format(y.shape)
    np.save('samples.npy', X)
    np.save('labels.npy', y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    shuffled_images = X[indices]
    shuffled_labels = y[indices]
    split = int(X.shape[0]*.9)
    training_images = shuffled_images[:split]
    training_labels = shuffled_labels[:split]

    val_images = shuffled_images[split:]
    val_labels = shuffled_labels[split:]

    return training_images, training_labels, val_images, val_labels

def resize_image_set(arr, target_size):
    # shuffle along the first axis
    np.random.shuffle(arr)
    # reduce size
    if len(arr) >= target_size:
        return arr[:target_size]
    # augment
    else:
        num_extra = target_size-len(arr)
        for i in range(num_extra):
            arr.append(arr[i])
    print len(arr)
    assert len(arr) == target_size
    return np.array(arr)









































    