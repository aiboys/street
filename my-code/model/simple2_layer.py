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

import cv2

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def simple2_layer(num_classes,input_shape):

      img_input = Input(input_shape, name='input')


      x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear', name='one')(img_input)
      x = BatchNormalization(name='BN_1')(x)


      x = Lambda(lambda x: tf.image.resize_images(x, [256,512]),name='lambda_1')(x)
      x = Activation('softmax', name='act')(x)




      model=Model(img_input, x)


      return model