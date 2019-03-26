# from keras.models import Model
# from keras.layers import Input
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# from keras.layers.core import Activation, Dropout, Lambda
# from keras.layers.normalization import BatchNormalization
# from keras.layers.merge import add, concatenate
# from keras.optimizers import Adam
# from keras import backend as K
#
# import tensorflow as tf
#
#
# def dice_coef(y_true, y_pred):
#     return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
#
# def pyramidconv(imginput,futures,filters,d_rates):
#     length_max=imginput.shape(0).value
#     img_input=Input(imginput[length_max])
#
#     x = Conv2D(filters, kernel_size=1, dilation_rate=d_rates)(img_input)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     for i in length_max:
#         temp=
#       concat=concatenate([imginput[i],x,futures])
#     imgorigin=imginput
#
#     return  x,concat
#
# def DenseASPP(num_classes, input_shape, lr_init, lr_decay):
#     img_input=Input(input_shape)
#
#     x = Conv2D(64, kernel_size=3, strides=(2, 2), padding='same')(img_input)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#
#     x = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
#
#     img=[x]
#     d_rate=[3,6,12,18,24]
#     filter=[128,64,256,512,1024]
#     future=[x]     #temp[0]是原始图 1- 0+3; 2- 0+3+6;  3- 0+3+6+12; 4- 0+3+6+12+18; 5- 0+3+6+12+18+24
#     concat=[x]
#     for i in range(1,5):
#         x,concat_img=pyramidconv(imginput=concat[:i-1],futures=future[:i-1],filters=filter[i-1],d_rates=d_rate[i-1])
#         future.append(x)                    #保存每一层的输出的卷积源图,同时也是下一层的输入图
#         concat.append(concat_img)           #保存每一层的concat图
#
#
#
#
