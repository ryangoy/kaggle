import cv2
import numpy as np
import os

from sklearn.model_selection import train_test_split

def load_data(valid_percent=.15, 
              fish_types=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'],
              fish_counts = [1745,202,117,68,442,286,177,740],
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
            for file in os.listdir("train_all/{}".format(fish)):
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
                X += [img]
                label = [0 for _ in range(len(fish_types))]
                label[i] = 1
                y += [label]
        X = np.array(X)
        y = np.array(y)
        print X.shape, y.shape
        np.save(savefileX, X)
        np.save(savefileY, y)
    else:
        X = np.load(savefileX)
        y = np.load(savefileY)
    # X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=.15)

    fish_cumulative_counts = [0] + [sum(fish_counts[:i+1]) for i in range(len(fish_counts))]
    nb_trn_all_samples = fish_cumulative_counts[-1]
    nb_trn_samples = int(sum([((1 - valid_percent)*100*c)//100 for c in fish_counts]))
    nb_val_samples = nb_trn_all_samples - nb_trn_samples

    X_trn = []
    X_val = []
    y_trn = []
    y_val = []
    for i in range(len(fish_counts)):
        Xt, Xv, yt, yv = train_test_split(X[fish_cumulative_counts[i]:fish_cumulative_counts[i+1]], 
                                          y[fish_cumulative_counts[i]:fish_cumulative_counts[i+1]], 
                                          test_size=valid_percent)
        X_trn += list(Xt)
        X_val += list(Xv)
        y_trn += list(yt)
        y_val += list(yv)
    X_trn = np.array(X_trn)
    X_val = np.array(X_val)
    y_trn = np.array(y_trn)
    y_val = np.array(y_val)
    # print X_trn.shape, y_trn.shape, X_val.shape, y_val.shape
    # print np.sum(y_trn, axis=0), np.sum(y_val, axis=0)
    # exit(1)
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
















