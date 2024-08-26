#optimized
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
import os

# This is for 3D
h = 720
w = 1280

# number of frames
n = 5

def load_data(path):
    images = sorted(glob(os.path.join(path, "*.jpeg")))
    x = [images[i:i+n] for i in range(len(images)-n)]
    y = images[n:]
    return x, y

# w, h
def read_image(path):
    x = cv2.imread(path.decode(), cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (w, h))
    x = x / 255.0
    return x

# tf_parse for 3D
def tf_parse(x, y):
    def _parse(x, y):
        X = [np.transpose(read_image(i), (2, 0, 1)) for i in x]
        X = np.stack(X, axis=-1)
        y = np.transpose(read_image(y), (2, 0, 1))[:, :, :, None]
        return X, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([3, h, w, n])
    y.set_shape([3, h, w, 1])

    return x, y

def tf_dataset(x, y, batch, train):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    if train:
        dataset = dataset.take(len(x) // 10)
    else:
        dataset = dataset.skip(len(x) // 10)
    dataset = dataset.batch(batch)
    return dataset
