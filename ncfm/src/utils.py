import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split

def load_data(valid_percent=.15, 
              fish_types=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'],
              fish_counts = [1745,202,117,68,442,286,177,740],
              fish_multipliers = [1,1,1,1,1,1,1,1],
              size=(270,480),
              saved=False,
              savefileX='X_default.npy',
              savefileY='y_default.npy'):
    if not saved:
        X = []
        y = []
        index = 0
        for i in range(len(fish_types)):
            fish = fish_types[i]
            # for file in os.listdir("preprocessed_train/{}".format(fish)):
            for file in os.listdir("train_all/{}".format(fish)):
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
    # X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=.15)
    # fish_mult_counts = [fish_counts[i] * fish_multipliers[i] for i in range(len(fish_counts))]
    fish_cumulative_counts = [0] + [sum(fish_counts[:i+1]) for i in range(len(fish_counts))]
    nb_trn_all_samples = fish_cumulative_counts[-1]
    trn_samples_counts = [((1 - valid_percent)*100*c)//100 for c in fish_counts]
    nb_val_samples = nb_trn_all_samples - int(sum(trn_samples_counts))
    nb_trn_samples = int(sum([trn_samples_counts[i] * fish_multipliers[i] for i in range(len(fish_multipliers))]))

    X_trn = []
    X_val = []
    y_trn = []
    y_val = []
    for i in range(len(fish_counts)):
        Xt, Xv, yt, yv = train_test_split(X[fish_cumulative_counts[i]:fish_cumulative_counts[i+1]], 
                                          y[fish_cumulative_counts[i]:fish_cumulative_counts[i+1]], 
                                          test_size=valid_percent)

        for _ in range(fish_multipliers[i]):
            X_trn += list(Xt)
            y_trn += list(yt)
        X_val += list(Xv)
        y_val += list(yv)
    X_trn = np.array(X_trn)
    X_val = np.array(X_val)
    y_trn = np.array(y_trn)
    y_val = np.array(y_val)
    # print X_trn.shape, y_trn.shape, X_val.shape, y_val.shape
    # print nb_trn_all_samples, nb_trn_samples, nb_val_samples
    # print np.sum(y_trn, axis=0), np.sum(y_val, axis=0)
    # exit(1)
    # USE MULTIPLIER WITH X and y
    return X, X_trn, X_val, y, y_trn, y_val

def load_data_bbox_2point(valid_percent=.15, 
              fish_types=['ALB','BET','DOL','LAG','OTHER','SHARK','YFT'],
              fish_counts = [1745,202,117,68,286,177,740],
              fish_multipliers = [1,1,1,1,1,1,1],
              size=(270,480),
              saved=False,
              savefileX='X_default.npy',
              savefileY='y_default.npy',
              output='regression'):
    if not saved:
        X = []
        y = []

        annotation_files = ['alb_labels.json',
                            'bet_labels.json',
                            'dol_labels.json',
                            'lag_labels.json',
                            'other_labels.json',
                            'shark_labels.json',
                            'yft_labels.json']
        name_to_2point = {}
        # no_annotation = 0
        for file in annotation_files:
            with open('annotations/{}'.format(file)) as f:
                annotations = json.load(f)
            for fish in annotations:
                if len(fish['annotations']) != 2:
                    # no_annotation += 1
                    continue
                points = [fish['annotations'][0]['x'], fish['annotations'][0]['y'], 
                         fish['annotations'][1]['x'], fish['annotations'][1]['y']]
                name_to_2point[fish['filename']] = points
            # print no_annotation

        # unused = 0
        # used = 0
        index = 0
        for i in range(len(fish_types)):
            fish = fish_types[i]
            for file in os.listdir("train_all/{}".format(fish)):
                path = "train_all/{}/{}".format(fish, file)
                img = cv2.imread(path)
                original_shape = img.shape
                # plt.figure()
                # plt.imshow(img)
                # plt.scatter(name_to_2point[file][0], name_to_2point[file][1], s=25, c='green', marker='o')
                # plt.scatter(name_to_2point[file][2], name_to_2point[file][3], s=25, c='red', marker='o')
                # plt.show()
                img = cv2.resize(img, (size[1], size[0]), cv2.INTER_LINEAR)
                img = img.transpose((2, 0, 1))
                if file in name_to_2point:
                    # used += 1
                    X += [img]
                    if output == 'regression':
                        y += [name_to_2point[file]]
                    elif output == 'classification':
                        x1c = int((name_to_2point[file][0] * size[1]) // original_shape[1])
                        y1c = int((name_to_2point[file][1] * size[0]) // original_shape[0])
                        x2c = int((name_to_2point[file][2] * size[1]) // original_shape[1])
                        y2c = int((name_to_2point[file][3] * size[0]) // original_shape[0])
                        # print x1c, y1c, x2c, y2c
                        # plt.figure()
                        # plt.imshow(img.transpose(1, 2, 0))
                        # plt.scatter(x1c, y1c, s=25, c='green', marker='o')
                        # plt.scatter(x2c, y2c, s=25, c='red', marker='o')
                        # plt.show()

                        x1 = [0 for _ in range(size[1])]
                        x1[x1c] = 1
                        y1 = [0 for _ in range(size[0])]
                        y1[y1c] = 1
                        x2 = [0 for _ in range(size[1])]
                        x2[x2c] = 1
                        y2 = [0 for _ in range(size[0])]
                        y2[y2c] = 1

                        y += [x1+y1+x2+y2]
                        # exit(1)
                    else:
                        raise Exception('must chose between regression and classification')
                # else:
                    # unused += 1
            # print used, unused
        X = np.array(X)
        y = np.array(y)
        print X.shape, y.shape
        np.save('data_arrays/bb_data/{}'.format(savefileX), X)
        np.save('data_arrays/bb_data/{}'.format(savefileY), y)
        # exit(1)
    else:
        X = np.load('data_arrays/bb_data/{}'.format(savefileX))
        y = np.load('data_arrays/bb_data/{}'.format(savefileY))
    # X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=.15)
    # fish_mult_counts = [fish_counts[i] * fish_multipliers[i] for i in range(len(fish_counts))]
    fish_cumulative_counts = [0] + [sum(fish_counts[:i+1]) for i in range(len(fish_counts))]
    nb_trn_all_samples = fish_cumulative_counts[-1]
    trn_samples_counts = [((1 - valid_percent)*100*c)//100 for c in fish_counts]
    nb_val_samples = nb_trn_all_samples - int(sum(trn_samples_counts))
    nb_trn_samples = int(sum([trn_samples_counts[i] * fish_multipliers[i] for i in range(len(fish_multipliers))]))

    X_trn = []
    X_val = []
    y_trn = []
    y_val = []
    for i in range(len(fish_counts)):
        Xt, Xv, yt, yv = train_test_split(X[fish_cumulative_counts[i]:fish_cumulative_counts[i+1]], 
                                          y[fish_cumulative_counts[i]:fish_cumulative_counts[i+1]], 
                                          test_size=valid_percent)

        for _ in range(fish_multipliers[i]):
            X_trn += list(Xt)
            y_trn += list(yt)
        X_val += list(Xv)
        y_val += list(yv)
    X_trn = np.array(X_trn)
    X_val = np.array(X_val)
    y_trn = np.array(y_trn)
    y_val = np.array(y_val)
    # print X_trn.shape, y_trn.shape, X_val.shape, y_val.shape
    # print nb_trn_all_samples, nb_trn_samples, nb_val_samples
    # print np.sum(y_trn, axis=0), np.sum(y_val, axis=0)
    # exit(1)
    # USE MULTIPLIER WITH X and y
    return X, X_trn, X_val, y, y_trn, y_val

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
















