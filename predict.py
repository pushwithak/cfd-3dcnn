#Optimized

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tqdm import tqdm
from data import load_data, read_image
from sklearn.model_selection import train_test_split

#number of frames
n = 12
def load_data(path):
    images = sorted(glob(os.path.join(path, "*.jpeg")))
    x = [images[i:i+n] for i in range(len(images)-n)]
    y = images[n:]
    return x, y


# w, h
def read_image(path):
    x = cv2.imread(path, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (1280, 720))
    x = x / 255.0
    return x

def parse(x, y):
    X = []
    for i in x:
        X.append(np.transpose(read_image(i), (2, 0, 1)))
    X = np.stack(X, axis=-1)
    y = np.transpose(read_image(y), (2, 0, 1))[:, :, :, None]
    return X, y


if __name__ == "__main__":
    path = "masked_frames_500"
    
    batch_size = 4

    X, Y = load_data(path)
    
    X = train_test_split(X, test_size=len(X) // 10, random_state = 42)
    Y = train_test_split(Y, test_size=len(Y) // 10, random_state = 42)
    X = X[1]
    Y = Y[1]
    
    """
    
    def mse_loss(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    def ssim_loss(y_true, y_pred):
        return 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 255))
    
    def ssim_metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 255))
    
    def combined_loss(y_true, y_pred):
        alpha = 0.5  # you can adjust this parameter
        return alpha * mse_loss(y_true, y_pred) + (1 - alpha) * ssim_loss(y_true, y_pred)
        
        """
    
    def ssim_metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    
    def mse_ssim_loss(y_true, y_pred, weightage = 0.5):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        ssim = tf.reduce_mean(1.0 - tf.image.ssim(y_true, y_pred, 1.0))
        return weightage * mse + (1.0 - weightage) * ssim

    model = tf.keras.models.load_model("files/model_3d_frames_12.h5", custom_objects={'mse_ssim_loss': mse_ssim_loss, 'ssim_metric': ssim_metric})
    
    #model = tf.keras.models.load_model("files/model_3d_frames_12_Exp_4_2_1.h5")

    for i, (x, y) in tqdm(enumerate(zip(X, Y))):
        x, y = parse(x, y)

        a_1 = str(Y[i])
        name = a_1.split("/")[-1].split(".")[0]

        y_pred = model.predict(np.expand_dims(x, axis=0))
        _, h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0

        y = np.transpose(y[:, :, :, 0], (1,2,0))
        y_pred = np.transpose(y_pred[0, :, :, :, 0], (1,2,0))

        all_images = [
            #y * 255.0, white_line,
            y_pred * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results_ssim/{name}.jpeg", image)