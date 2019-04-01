from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils import plot_model
def dice_coef(y_true, y_pred):


    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
      #这是对的


# # return K.mean(2.*(K.sum(y_true * y_pred)) / (K.sum(y_true*y_true) + K.sum(y_pred*y_pred)))     #dice_coef
# return 2. * K.sum(y_pred * y_true) / (K.sum(y_true) + K.sum(y_pred))     # MIoU

def fcn2_shallow( input_shape):  #64  128 256

    img_input = Input(input_shape, name='input')

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = MaxPooling2D()(x)
    # 64 128 128
    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    model = Model(img_input, x)

    return model
