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


def with_cnn_deep(num_classes, input_shape):

    img_input = Input(input_shape)

#----------------------------------------------------------------------------------------------

    x = Lambda(lambda x: tf.image.resize_images(x, [128, 256]), name='labm2')(img_input)
    # x=MaxPooling2D(name='pool1')(x)

    x = Conv2D(256, (3, 3),padding='same', name='block3_conv3_')(x)
    x = BatchNormalization(name='bn8')(x)
    x = Activation('relu',name='act8')(x)
    # x = Lambda(lambda x: tf.image.resize_images(x, [64, 128]))(x)

    block_3_out = MaxPooling2D(name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1_')(block_3_out)
    x = BatchNormalization(name='bn9')(x)
    x = Activation('relu',name='act9')(x)


    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2_')(x)
    x = BatchNormalization(name='bn10')(x)
    x = Activation('relu',name='act10')(x)
    #
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3_')(x)
    x = BatchNormalization(name='bn11')(x)
    x = Activation('relu',name='act11')(x)
    #
    block_4_out = MaxPooling2D(name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1_')(block_4_out)
    x = BatchNormalization(name='bn12')(x)
    x = Activation('relu',name='act12')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2_')(x)
    x = BatchNormalization(name='bn13')(x)
    x = Activation('relu',name='act13')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3_')(x)
    x = BatchNormalization(name='bn14')(x)
    x = Activation('relu',name='act14')(x)

    x = MaxPooling2D(name='pool5')(x)
    #
    # # Load pretrained weights.
    # # if vgg_weight_path is not None:
    # #     vgg16 = Model(img_input, x)
    # #     vgg16.load_weights(vgg_weight_path, by_name=True)
    #
    # # Convolutinalized fully connected layer.
    x = Conv2D(4096, (7, 7), activation='relu', padding='same',name='123')(x)
    x = BatchNormalization(name='bn15')(x)
    x = Activation('relu',name='act15')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same',name='1234')(x)
    x = BatchNormalization(name='bn16')(x)
    x = Activation('relu',name='act16')(x)
    #
    # # Classifying layers.
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',name='12345')(x)
    x = BatchNormalization(name='bn17')(x)
    #
    block_3_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',name='123456')(block_3_out)
    block_3_out = BatchNormalization(name='bn18')(block_3_out)
    #
    block_4_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',name='1234567')(block_4_out)
    block_4_out = BatchNormalization(name='bn19')(block_4_out)
    #
    x = Lambda(lambda x: tf.image.resize_images(x, [32,64]))(x)
    x = Add()([x, block_4_out])
    x = Activation('relu',name='act17')(x)
    #
    x = Lambda(lambda x: tf.image.resize_images(x, [64,128]))(x)
    x = Add()([x, block_3_out])
    x = Activation('relu',name='act18')(x)
    #
    x = Lambda(lambda x: tf.image.resize_images(x, [512,1024]))(x)

    x = Activation('softmax',name='act19')(x)

    model = Model(img_input, x)
    return model
