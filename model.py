import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def conv2d_block(input_tensor, n_filters, kernel_size = 3):

    x = input_tensor

    for i in range(2):
        x = Conv2D(n_filters, kernel_size=(kernel_size, kernel_size), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
    return x

def encoding_block(inputs, n_filters=64,  pool_size=(2,2), dropout=0):
    f = conv2d_block(inputs, n_filters=n_filters)
    p = MaxPooling2D(pool_size=(2, 2))(f)
    p = Dropout(0)(p)

    return f, p

def encoder(inputs):
    f1, p1 = encoding_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0)
    f2, p2 = encoding_block(p1, n_filters=128, pool_size=(2, 2), dropout=0)
    f3, p3 = encoding_block(p2, n_filters=256, pool_size=(2, 2), dropout=0)
    #f4, p4 = encoding_block(p3, n_filters=512, pool_size=(2, 2), dropout=0)

    return p3, (f1, f2, f3)

def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=512)

    return bottle_neck

def decoder_block(inputs, conv_output, n_filters, kernel_size=3, strides=(2,2), dropout=0.0):
    u = Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    c = concatenate([u, conv_output])
    c = Dropout(dropout)(c)
    c = conv2d_block(c, n_filters=n_filters, kernel_size=3)

    return c
def decoder(inputs, convs):
    f1, f2, f3 = convs
    #c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=(3, 3), strides=(2, 2), dropout=0)
    c7 = decoder_block(inputs, f3, n_filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0)

    output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(c9)

    return output

def unet():
    inputs = Input(shape=(128, 128, 3))
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs)
    model = Model(inputs, outputs)

    return model


if __name__ == "__main__":
    model = unet()
    model.summary()
