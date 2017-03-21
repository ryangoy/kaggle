import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from os.path import join

dataset_dir = '/home/ryan/cs/datasets/ncfm'
model_path = join(dataset_dir, 'classifier_weights.h5')
output_path = join(dataset_dir, 'final_classified.npy')
aligned_output = join(dataset_dir, 'aligned_output.npy')

# output of the aligner
images = np.load(aligned_output)
images -= images.mean(axis=(1,2),keepdims=1)
images *= 255
images = images[:, :, :, ::-1]

model = load_model(model_path)
predictions = model.predict(images)

print "Prediction's checksum: {}".format(np.sum(predictions, axis=0))

test_images = np.load(aligned_output)

for i in range(100):
    print predictions[i]
    plt.imshow(test_images[i])
    plt.show()

np.save(output_path, predictions)