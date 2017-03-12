import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# large prime for modding
prime = 10007

def split_k_folds(k=3, fold_file=None):
    # images with no bbox annotation
    # for fish in ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']:
    #     l1 = sorted(os.listdir("train_all/{}".format(fish)))
    #     l2 = sorted(os.listdir("train_all_cropped2/{}".format(fish)))
    #     print sorted(list(set(l1) - set(l2)))
    # exit(1)
    if fold_file != None and os.path.isfile(fold_file):
        with open(fold_file, 'r') as f:
            info = json.load(f)
            file2set = info['file2set']
            checksums = info['checksums']
            file_folds = info['file_folds']
    else:
        file_folds = [[] for _ in range(k)]
        index = 0
        # counts = [0 for _ in range(k)]
        checksums = [0 for _ in range(k)]
        for fish in ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']:
            files = sorted(os.listdir("train_all_cropped2/{}".format(fish)))
            np.random.shuffle(files)
            for f in files:
                file_folds[index] += [f]
                # file2set[f] = index
                # counts[index] += 1
                index = (index + 1) % k
            # print counts
        file2set = {}
        for i in range(len(file_folds)):
            file_folds[i] = sorted(file_folds[i])
            # print file_folds[i][:10]
            count = 0
            for file in file_folds[i]:
                file2set[file] = i
                checksums[i] = (checksums[i] + int(file[5:9]))**2 % prime
        # print len(file2set)
        # print fold_file
        if fold_file != None:
            info = {}
            info['file2set'] = file2set
            info['checksums'] = checksums
            info['file_folds'] = file_folds
            with open(fold_file, 'w') as f:
                json.dump(info, f)
    print checksums
    return file2set, checksums, file_folds


def load_data(fold_file=None,
              fish_types=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'],
              size=(270,480),
              saved=False,
              savefileX='X_default.npy',
              savefileY='y_default.npy',
              k=3):
    if not saved:
        X = []
        y = []
        index = 0
        for i in range(len(fish_types)):
            fish = fish_types[i]
            # for file in os.listdir("preprocessed_train/{}".format(fish)):
            for file in sorted(os.listdir("train_all/{}".format(fish))):
                # path = "preprocessed_train/{}/{}".format(fish, file)
                path = "train_all/{}/{}".format(fish, file)
                # img = np.array(keras.preprocessing.image.load_img(path, target_size=vgg_size))
                # img = skimage.io.imread(path)
                img = cv2.imread(path)
                # img = skimage.transform.resize(img, vgg_size).transpose((2, 0, 1))
                # print img.shape
                img = cv2.resize(img, (size[1], size[0]), cv2.INTER_LINEAR)
                # print img.shape
                img = img.transpose((2, 0, 1))
                # print img.shape
                label = [0 for _ in range(len(fish_types))]
                label[i] = 1
                X += [img]
                y += [label]
        X = np.array(X)
        y = np.array(y)
        print X.shape, y.shape
        np.save('data_arrays/e2e_data/{}'.format(savefileX), X)
        np.save('data_arrays/e2e_data/{}'.format(savefileY), y)
    else:
        X = np.load('data_arrays/e2e_data/{}'.format(savefileX))
        y = np.load('data_arrays/e2e_data/{}'.format(savefileY))

    X_folds = [[] for _ in range(k)]
    y_folds = [[] for _ in range(k)]
    filename_folds = [[] for _ in range(k)]

    files = []
    for i in range(len(fish_types)):
        fish = fish_types[i]
        files += os.listdir("train_all/{}".format(fish))
    files.sort()

    file2set, expected_checksums, file_folds = split_k_folds(k=k, fold_file=fold_file)

    checksums = [0 for _ in range(k)]

    for i in range(len(files)):
        if files[i] not in file2set:
            # print files[i]
            continue
        X_folds[file2set[files[i]]] += [list(X[i])]
        y_folds[file2set[files[i]]] += [list(y[i])]
        filename_folds[file2set[files[i]]] += [files[i]]
        checksums[file2set[files[i]]] = (checksums[file2set[files[i]]] + int(files[i][5:9]))**2 % prime
        # checksums[file2set[files[i]]] = (checksums[file2set[files[i]]] + int(files[i][5:9])**2) % prime
    for i in range(k):
        if filename_folds[i] != file_folds[i]:
            raise Exception('fold file orders do not match, make sure folds are the same')
    for i in range(k):
        X_folds[i] = np.array(X_folds[i])
        y_folds[i] = np.array(y_folds[i])
        print '[FOLD {}]'.format(i+1)
        print X_folds[i].shape, y_folds[i].shape
        print np.sum(y_folds[i], axis=0)
    if checksums != expected_checksums:
        print checksums
        raise Exception('checksums do not match, make sure folds are the same')
    print 'checksums match'
    # exit(1)
    return X, y, X_folds, y_folds, filename_folds


def load_data_cropped(fold_file=None,
              fish_types=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'],
              size=(270,480),
              saved=False,
              savefileX='X_default.npy',
              savefileY='y_default.npy',
              k=3):
    if not saved:
        X = []
        y = []
        index = 0
        for i in range(len(fish_types)):
            fish = fish_types[i]
            for file in sorted(os.listdir("train_all_cropped2/{}".format(fish))):
                path = "train_all_cropped2/{}/{}".format(fish, file)
                # img = np.array(keras.preprocessing.image.load_img(path, target_size=vgg_size))
                # img = skimage.io.imread(path)
                img = plt.imread(path)
                if img.shape[0] > img.shape[1]:
                    img = img.transpose((1, 0, 2))
                # plt.imshow(img)
                # plt.show()
                # avg = np.mean(np.mean(img, axis=0), axis=0)
                # if img.shape[0] > img.shape[1]:
                #     temp = np.zeros((img.shape[0], img.shape[0], 3)) + avg
                #     start_index = img.shape[0] / 2 - img.shape[1]/2
                #     temp[:,start_index:start_index+img.shape[1],:] = img
                # elif img.shape[0] < img.shape[1]:
                #     temp = np.zeros((img.shape[1], img.shape[1], 3)) + avg
                #     start_index = img.shape[1] / 2 - img.shape[0]/2
                #     temp[start_index:start_index+img.shape[0],:,:] = img
                # # plt.imshow(temp)
                # # plt.show()
                # img = temp
                # # img = skimage.transform.resize(img, vgg_size).transpose((2, 0, 1))
                # # print img.shape
                img = cv2.resize(img, (size[1], size[0]), cv2.INTER_LINEAR)
                # plt.imshow(img)
                # plt.show()
                # exit(1)
                # print img.shape
                img = img.transpose((2, 0, 1))
                # print img.shape
                X += [img]
                label = [0 for _ in range(len(fish_types))]
                label[i] = 1
                y += [label]
        X = np.array(X)
        y = np.array(y)
        print X.shape, y.shape
        np.save('data_arrays/cropped_data/{}'.format(savefileX), X)
        np.save('data_arrays/cropped_data/{}'.format(savefileY), y)
    else:
        X = np.load('data_arrays/cropped_data/{}'.format(savefileX))
        y = np.load('data_arrays/cropped_data/{}'.format(savefileY))
    
    X_folds = [[] for _ in range(k)]
    y_folds = [[] for _ in range(k)]
    filename_folds = [[] for _ in range(k)]

    files = []
    for i in range(len(fish_types)):
        fish = fish_types[i]
        files += os.listdir("train_all_cropped2/{}".format(fish))
    files.sort()

    file2set, expected_checksums, file_folds = split_k_folds(k=k, fold_file=fold_file)

    checksums = [0 for _ in range(k)]

    for i in range(len(files)):
        if files[i] not in file2set:
            # print files[i]
            continue
        X_folds[file2set[files[i]]] += [list(X[i])]
        y_folds[file2set[files[i]]] += [list(y[i])]
        filename_folds[file2set[files[i]]] += [files[i]]
        checksums[file2set[files[i]]] = (checksums[file2set[files[i]]] + int(files[i][5:9]))**2 % prime
        # checksums[file2set[files[i]]] = (checksums[file2set[files[i]]] + int(files[i][5:9])**2) % prime
    for i in range(k):
        if filename_folds[i] != file_folds[i]:
            raise Exception('fold file orders do not match, make sure folds are the same')
    for i in range(k):
        X_folds[i] = np.array(X_folds[i])
        y_folds[i] = np.array(y_folds[i])
        print '[FOLD {}]'.format(i+1)
        print X_folds[i].shape, y_folds[i].shape
        print np.sum(y_folds[i], axis=0)
    if checksums != expected_checksums:
        print checksums
        raise Exception('checksums do not match, make sure folds are the same')
    print 'checksums match'
    # exit(1)
    return X, y, X_folds, y_folds, filename_folds

def load_data_aligned(fold_file=None,
              fish_types=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'],
              size=(224,224),
              saved=False,
              savefileX='X_default.npy',
              savefileY='y_default.npy',
              k=3):

    X = []
    y = []
    index = 0
    for i in range(len(fish_types)):
        fish = fish_types[i]
        for file in sorted(os.listdir("train_all_cropped2/{}".format(fish))):
            path = "train_all_cropped2/{}/{}".format(fish, file)
            img = plt.imread(path)
            if img.shape[0] > img.shape[1]:
                img = img.transpose((1, 0, 2))
            img = cv2.resize(img, (size[1], size[0]), cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            X += [img]
            label = [0 for _ in range(len(fish_types))]
            label[i] = 1
            y += [label]
    X = np.array(X)
    y = np.array(y)
    print X.shape, y.shape
    
    X_folds = [[] for _ in range(k)]
    y_folds = [[] for _ in range(k)]
    filename_folds = [[] for _ in range(k)]

    files = []
    for i in range(len(fish_types)):
        fish = fish_types[i]
        files += os.listdir("train_all_cropped2/{}".format(fish))
    files.sort()

    file2set, expected_checksums, file_folds = split_k_folds(k=k, fold_file=fold_file)

    checksums = [0 for _ in range(k)]

    for i in range(len(files)):
        if files[i] not in file2set:
            # print files[i]
            continue
        X_folds[file2set[files[i]]] += [list(X[i])]
        y_folds[file2set[files[i]]] += [list(y[i])]
        filename_folds[file2set[files[i]]] += [files[i]]
        checksums[file2set[files[i]]] = (checksums[file2set[files[i]]] + int(files[i][5:9]))**2 % prime
        # checksums[file2set[files[i]]] = (checksums[file2set[files[i]]] + int(files[i][5:9])**2) % prime
    for i in range(k):
        if filename_folds[i] != file_folds[i]:
            raise Exception('fold file orders do not match, make sure folds are the same')
    for i in range(k):
        X_folds[i] = np.array(X_folds[i])
        y_folds[i] = np.array(y_folds[i])
        print '[FOLD {}]'.format(i+1)
        print X_folds[i].shape, y_folds[i].shape
        print np.sum(y_folds[i], axis=0)
    if checksums != expected_checksums:
        print checksums
        raise Exception('checksums do not match, make sure folds are the same')
    print 'checksums match'
    # exit(1)
    return X, y, X_folds, y_folds, filename_folds


def write_submission(filenames, predfile='default.npy', subfile='default.csv'):
    predictions_full = np.load("pred/{}".format(predfile))
    preds = np.clip(predictions_full,0.02, 1.00, out=None)
    # preds = np.zeros((nb_test_samples, nb_classes)) + .0414
    # for i in range(predictions_full.shape[0]):
    #     preds[i][np.argmax(predictions_full[i])] = .71

    with open('submissions/{}'.format(subfile), 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % (p/np.sum(preds[i, :])) for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")
















