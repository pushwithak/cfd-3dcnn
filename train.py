import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
from model import build_model

def write_metrics_to_file(history, file_name):
    with open(file_name, 'w') as f:
        f.write("Epoch, Train Accuracy, Val Accuracy, Train Loss, Val Loss\n")
        for epoch in range(len(history.history['accuracy'])):
            f.write(f"{epoch + 1}, {history.history['accuracy'][epoch]}, {history.history['val_accuracy'][epoch]}, {history.history['loss'][epoch]}, {history.history['val_loss'][epoch]}\n")

if __name__ == "__main__":
    
    ## Dataset
    path = "masked_frames_500"
    x,y = load_data(path)
    # print(len(x))
    # print(len(y))

    ## Hyperparameters
    epochs = 5
    batch = 2
    lr = 1e-1
    
    train_dataset = tf_dataset(x, y, batch=batch, train = True)
    # print(len(train_dataset))
    
    valid_dataset = tf_dataset(x, y, batch=batch, train = False)
    # print(len(valid_dataset))
    
    model = build_model()

    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["accuracy"]
    model.compile(loss="mse", optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint("files/model_3d.h5"),
        # TensorBoard(log_dir="./graph", histogram_freq=0, write_graph=True, write_images=True),
        EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="auto")
    ]

    history = model.fit(train_dataset,
              epochs=epochs,
              callbacks=callbacks,
              validation_data = valid_dataset
              )
    
    history.history
    
    print(history.history.keys())
    
    # Save metrics to a text file
    write_metrics_to_file(history, "model_history_batch_norm.txt")
    
    #  "Accuracy"
    plt.plot(history.history['accuracy'], '-x')
    plt.plot(history.history['val_accuracy'], '-x')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.savefig("accuracy_3D.png")
    plt.close()
    
    # "Loss"
    plt.plot(history.history['loss'], '-x')
    plt.plot(history.history['val_loss'], '-x')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.savefig("loss_3D.png")
    plt.close()