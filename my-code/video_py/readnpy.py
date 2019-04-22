import keras.backend as K
from keras.layers.core import Activation, Dropout, Lambda
import numpy as np
import random
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

def pre_processing(img):
    #随机增强0.9~1.1倍

    return img / 127.5 - 1  #归一化为-1:1


# Get ImageDataGenerator arguments(options) depends on mode - (train, val, test)
def get_data_gen_args():

        x_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)


        return x_data_gen_args, y_data_gen_args




# Data generator for fit_generator.


def data_generator_cnn(b_size):

    # f1 = np.load('deep_feature_2.npy')       # 512 256 512 19
    # deep_feature = np.squeeze(f1)            # 512 256 512 19
    # deep_feature = deep_feature[:,:,:,:]
    shallow_feature = np.load('shallow_feature_2.npy')  #
    # f2=np.load('flow_1.npy')               # 512 256 512 19
    # shallow_feature = np.squeeze(f2)

    # label = np.load('label_2.npy')           #512 256 512

    x_imgs = shallow_feature[:shallow_feature.shape[0]-1,:,:,:]        #original_data
    y_imgs= shallow_feature[1:,:,:,:]        #
    # y_imgs = label[:,:,:]        #

    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args()
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = x_imgs.shape[0]  #511
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]  #0....511随机数
            # c=x_imgs[idx]
            # cc=y_imgs[idx]
            x.append(x_imgs[idx])
            y.append(y_imgs[idx])

            if len(x) == b_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(b_size)
                seed = random.randrange(1, 100)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)

                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, y_result

                x.clear()
                y.clear()

