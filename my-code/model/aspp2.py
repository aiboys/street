from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
import cv2

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def ASPP2(num_classes, input_shape, lr_init, lr_decay):
    img_input=Input(input_shape)
    # t_start = cv2.getTickCount()
    x = Conv2D(32, kernel_size=3, strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=3, strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)   # 64 128 64

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)    #32 64 256

   #--------------------------------------------------------------------------------------------

    dil_0=x #32 64 256
    #渐层网络时间
    # t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    # print("渐层 Time: %.3f ms" % t_total)                   # 230ms

    x = Conv2D(256, kernel_size=3, dilation_rate=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    dil_1=x    #32 64 256
    concat_1=concatenate([dil_0,x])  #32 64 512

    x=Conv2D(512,kernel_size=3,dilation_rate=6, padding="same")(concat_1)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    dil_2=x
    concta_2=concatenate([dil_0,dil_1,x])   #32 64 1024

    x=Conv2D(512,kernel_size=3,dilation_rate=12, padding="same")(concta_2)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    dil_3=x  #32 64 512
    concat_3=concatenate([dil_0,dil_2,x])  # 32 64 1280

    x=Conv2D(512,kernel_size=3,dilation_rate=18, padding="same")(concat_3)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    dil_4=x  #32 64 512
    concat_4=concatenate([dil_0,dil_3,x]) #32 64 1280

    # x=Conv2D(1024,kernel_size=3,dilation_rate=24, padding="same")(concat_4)
    # x=BatchNormalization()(x)
    # x=Activation('relu')(x)
    # dil_5=x
    # concta_5=concatenate([dil_0,dil_4,x])

    x = Conv2D(1024, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1]*8, x.shape[2]*8)))(x)
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(x)
    x = BatchNormalization()(x)


    x = Activation('softmax')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])

    # plot_model(model, to_file='model_train_result/aspp_model.png')
    # plt.savefig('model_train_result/aspp_model.png')
    model.summary()

    return model








