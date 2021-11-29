import os
from glob import glob
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(path, split_size=0.1):

    images = sorted(glob(os.path.join(path, "Original/*")))
    masks = sorted(glob(os.path.join(path,  "Ground Truth/*")))

    total_size = len(images)
    test_size = int(total_size * split_size)
    valid_size = int(total_size * split_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_images(path) :
    path = path.decode(encoding='UTF-8', errors='strict')
    x = cv.imread(path, cv.IMREAD_COLOR)
    x = cv.resize(x, (128, 128))
    x = x / 255.0
    return x

def read_masks(path) :
    path = path.decode(encoding='UTF-8', errors='strict')
    x = cv.imread(path, cv.IMREAD_GRAYSCALE)
    x = cv.resize(x, (128, 128))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x

def  tf_parse(x, y):
    def parse(x, y):
        x = read_images(x)
        y = read_masks(y)
        return x, y

    x, y = tf.numpy_function(parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([128, 128, 3])
    y.set_shape([128, 128, 1])

    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


if __name__ == "__main__":

    print("")
    path = "Polyp_Dataset\PNG"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    print(test_x[1:2])
    ds =tf_dataset(train_x, train_y)
    for x, y in ds:
        print(x.shape, y.shape)
        break





