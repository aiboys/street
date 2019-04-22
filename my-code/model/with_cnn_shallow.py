from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
from keras.utils import plot_model

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def with_cnn_shallow(input_shape):
    img_input = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1_')(img_input)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu',name='act')(x)


    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2_')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu',name='act2')(x)

    x = MaxPooling2D(name='pool1')(x)


    # Block 2

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1_')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu',name='act3')(x)


    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2_')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu',name='act4')(x)

    x = MaxPooling2D(name='pool2')(x)

    # Block 3

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1_')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu',name='act5')(x)


    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2_')(x)
    x = BatchNormalization(name='bn6')(x)
    x = Activation('relu',name='act6')(x)
    # x = Lambda(lambda x: tf.image.resize_images(x, [256, 512]))(x)


    x=Conv2D(19, (1, 1), strides=(1, 1), activation='linear', name='shallow_point')(x)
    x = BatchNormalization(name='bn7')(x)
    x = Activation('relu',name='act7')(x)
    x = Lambda(lambda x: tf.image.resize_images(x, [512, 1024]), name='lam1')(x)

#----------------------------------------------------------------------------------------------


    model = Model(img_input, x)

    return model
