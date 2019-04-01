import h5py
import numpy as np
import random
import cv2
import csv
from keras.preprocessing.image import ImageDataGenerator



def pre_processing(img):
    #随机增强0.9~1.1倍
    rand_s = random.uniform(0.9, 1.1)
    rand_v = random.uniform(0.9, 1.1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


    return img / 127.5 - 1  #归一化为-1:1


# Get ImageDataGenerator arguments(options) depends on mode - (train, val, test)
def get_data_gen_args(mode):
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
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

    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=pre_processing)
        y_data_gen_args = dict()

    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args, y_data_gen_args


# One hot encoding for y_img.
def get_result_map(b_size, y_img):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, 256, 512, 9))

    # import pandas as pd
    # data1 = pd.DataFrame(y_img[:,:,0])
    # data1.to_csv('data1.csv')

    # For np.where calculation.
    person = (y_img == 24)
    car = (y_img == 26)
    road = (y_img == 7)
    rider=(y_img==24)
    truck=(y_img==27)
    bus=(y_img==28)
    motorcycle=(y_img==32)
    bicycle=(y_img==33)

    background = np.logical_not(person + car + road+rider+truck+bus+motorcycle+bicycle)
    # if person==1:
    #    print("person!\n")
    # for i in range(20):
    result_map[:, :, :, 0] = np.where(background, 1, 0)
    result_map[:, :, :, 1] = np.where(road, 1, 0)
    result_map[:, :, :, 2] = np.where(person, 1, 0)
    result_map[:, :, :, 3] = np.where(rider, 1, 0)
    result_map[:, :, :, 4] = np.where(car, 1, 0)
    result_map[:, :, :, 5] = np.where(truck, 1, 0)
    result_map[:, :, :, 6] = np.where(bus, 1, 0)
    result_map[:, :, :, 7] = np.where(motorcycle, 1, 0)
    result_map[:, :, :, 8] = np.where(bicycle, 1, 0)

    return result_map


# Data generator for fit_generator.
def data_generator_video(d_path, b_size, mode):
    data = h5py.File(d_path, 'r')
    x_imgs = data.get('/' + mode + '/x')   #original_data
    y_imgs = data.get('/' + mode + '/y')   #truth_data

    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = x_imgs.shape[0]  #512
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]

            x.append(x_imgs[idx].reshape((256, 512, 3)))
            y.append(y_imgs[idx].reshape((256, 512, 1)))

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

                yield x_result, get_result_map(b_size, y_result)

                x.clear()
                y.clear()
