#from typing import Concatenate
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model

n = 5

def build_model():
    
    input_layer = Input((3, 720, 1280, n))
    
    # Define the 3D CNN layers
    x = tf.keras.layers.Conv3D(8, kernel_size=(3,3,3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
   
    x = tf.keras.layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv3D(filters=1, kernel_size=(3,3,3), activation='sigmoid', padding='same')(x)

    return Model(inputs=input_layer, outputs = x)

if __name__ == "__main__":
    model = build_model()
    model.summary()