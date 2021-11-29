import os
from glob import glob
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from data import load_data, tf_dataset
from model import unet

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true*y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15)/ (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


if __name__ == "__main__":

    print("")
    path = "Polyp_Dataset\PNG"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    ##hyper-parameter
    batch = 4
    lr = 1e-4
    epochs = 50

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    model = unet()

    optimizer = Adam(learning_rate=lr)

    metrics = ['acc', Recall(), Precision(), iou]
    model.compile(loss=BinaryCrossentropy(), optimizer=optimizer, metrics=metrics)

    callbacks = [
        ModelCheckpoint("files/model.h5"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps +=1

    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )



