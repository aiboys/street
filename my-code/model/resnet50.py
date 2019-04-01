import numpy as np


import keras.backend as K
from keras import layers
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda, Activation
import tensorflow as tf
def dice_coef(y_true, y_pred):


    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
      #这是对的

def resnet(input_shape,num_classes, lr_init, lr_decay,):

    # classes = 1000


    input_data = Input(shape=input_shape,name='input')

    #stage1------------------------------------------
    x = ZeroPadding2D((3, 3))(input_data)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)


    #stage2 -----------------------------------------
    x = conv_block(base_name='2a',input_layer=x,filters=[64, 64, 256],kernal_size=3,stride=(1, 1))   #63 127 256
    x = identity_block(base_name='2b',input_layer=x, kernal_size=3, filters=[64, 64, 256])
    x = identity_block(base_name='2c',input_layer=x, kernal_size=3, filters=[64, 64, 256])

    #stage3 -----------------------------------------
    x = conv_block(base_name='3a',input_layer=x,filters=[128, 128, 512],kernal_size=3,stride=(2, 2))  #32 64 512
    x = identity_block(base_name='3b',input_layer=x, kernal_size=3, filters=[128, 128, 512])
    x = identity_block(base_name='3c',input_layer=x, kernal_size=3, filters=[128, 128, 512])
    x = identity_block(base_name='3d',input_layer=x, kernal_size=3, filters=[128, 128, 512])
    x = Lambda(lambda x: tf.image.resize_images(x, [64, 128]))(x)

    #stage4 -----------------------------------------
    x = conv_block(base_name='4a',input_layer=x,filters=[256, 256, 1024],kernal_size=3,stride=(1, 1))   #64 128 1024
    x = identity_block(base_name='4b',input_layer=x, kernal_size=3, filters=[256, 256, 1024])
    x = identity_block(base_name='4c',input_layer=x, kernal_size=3, filters=[256, 256, 1024])
    x = identity_block(base_name='4d',input_layer=x, kernal_size=3, filters=[256, 256, 1024])
    x = identity_block(base_name='4e',input_layer=x, kernal_size=3, filters=[256, 256, 1024])
    x = identity_block(base_name='4f',input_layer=x, kernal_size=3, filters=[256, 256, 1024])
    x = Lambda(lambda x: tf.image.resize_images(x, [128,256]))(x)
    #stage5 -----------------------------------------
    x = conv_block(base_name='5a',input_layer=x,filters=[512, 512, 2048],kernal_size=3,stride=(1, 1))
    x = identity_block(base_name='5b',input_layer=x, kernal_size=3, filters=[512, 512, 2048])
    x = identity_block(base_name='5c',input_layer=x, kernal_size=3, filters=[512, 512, 2048])   #128 256 2048
    x=Lambda(lambda x: tf.image.resize_images(x, [256,512]))(x)


    # x = Flatten()(x)
    # x = Dense(num_classes, activation='softmax', name='fc1000')(x)

    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)
    inputs = input_data

    model = Model(inputs, x, name='resnet50')
    # model.load_weights( weight_dir,by_name=True)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    model.summary()
    return model


def identity_block(base_name,input_layer,kernal_size,filters):


    #比如 input:(batch,h,w,256), filters=[64,64,256]

    # (1,1,64)
    x = Conv2D(filters[0],(1,1),name="res"+base_name+'_branch2a')(input_layer)
    x = BatchNormalization(axis=3,name="bn"+base_name+'_branch2a')(x)
    x = Activation('relu')(x)
    # --> (batch,h,w,64)

    # (3,3,64)
    x = Conv2D(filters[1],kernal_size,padding='same',name="res"+base_name+'_branch2b')(x)
    x = BatchNormalization(axis=3,name="bn"+base_name+'_branch2b')(x)
    x = Activation('relu')(x)
    # --> (batch,h,w,64)

    # (1,1,256)
    x = Conv2D(filters[2],(1,1),name="res"+base_name+'_branch2c')(x)
    x = BatchNormalization(axis=3,name="bn"+base_name+'_branch2c')(x)
    # --> (batch,h,w,256)

    x = layers.add([x, input_layer])
    x = Activation('relu')(x)
    # --> (batch,h,w,256)

    return x

def conv_block(base_name,input_layer,kernal_size,filters,stride=(2,2)):

    #比如 input:(batch,32,32,256),filters = [64,64,256]

    # (1,1,64) /2
    x = Conv2D(filters[0], (1, 1), strides=stride ,name="res"+base_name+"_branch2a")(input_layer)
    x = BatchNormalization(axis=3, name="bn" + base_name+"_branch2a")(x)
    x = Activation('relu')(x)
    # --> (batch,16,16,64)

    # (3,3,64)
    x = Conv2D(filters[1], kernal_size,padding='same', name="res" + base_name+'_branch2b')(x)
    x = BatchNormalization(axis=3, name="bn" + base_name+'_branch2b')(x)
    x = Activation('relu')(x)
    # --> (batch,h,w,64)

    # (1,1,256)
    x = Conv2D(filters[2], (1, 1), name="res" + base_name+'_branch2c')(x)
    x = BatchNormalization(axis=3, name="bn" + base_name+'_branch2c')(x)
    # --> (batch,h,w,256)

    # (1,1,256) /2
    shortcut = Conv2D(filters[2],(1, 1),strides=stride,name="res"+base_name+'1')(input_layer)
    shortcut = BatchNormalization(axis=3,name="bn"+base_name+'1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)

    return x
