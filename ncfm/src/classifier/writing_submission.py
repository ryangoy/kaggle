import os
import sys
import numpy as np

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


if __name__ == '__main__':
    write_submission()
