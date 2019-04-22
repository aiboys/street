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


# One hot encoding for y_img.
def get_result_map(b_size, y_img):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, 256, 512, 19,1))

    # import pandas as pd
    # data1 = pd.DataFrame(y_img[:,:,0])
    # data1.to_csv('data1.csv')

    # For np.where calculation.
    person = (y_img == 24)
    car = (y_img == 26)
    road = (y_img == 7)
    sidewalk=(y_img==8)
    building=(y_img==11)
    wall=(y_img==12)
    fence=(y_img==13)
    pole=(y_img==17)
    trafficLight=(y_img==19)
    trafficSign=(y_img==20)
    vegetation=(y_img==21)
    ground=(y_img==6)
    sky=(y_img==23)
    rider=(y_img==24)
    truck=(y_img==27)
    bus=(y_img==28)
    motorcycle=(y_img==32)
    bicycle=(y_img==33)

    background = np.logical_not(person + car + road+sidewalk+building+ wall+ fence+pole+trafficLight+trafficSign+vegetation+ground+sky+rider+truck+bus+motorcycle+bicycle)
    # if person==1:
    #    print("person!\n")
    # for i in range(20):
    result_map[:, :, :, 0,:] = np.where(background, 1, 0)
    result_map[:, :, :, 1,:] = np.where(road, 1, 0)
    result_map[:, :, :, 2,:] = np.where(sidewalk, 1, 0)
    result_map[:, :, :, 3,:] = np.where(building, 1, 0)
    result_map[:, :, :, 4,:] = np.where(wall, 1, 0)
    result_map[:, :, :, 5,:] = np.where(fence, 1, 0)
    result_map[:, :, :, 6,:] = np.where(pole, 1, 0)
    result_map[:, :, :, 7,:] = np.where(trafficLight, 1, 0)
    result_map[:, :, :, 8,:] = np.where(trafficSign, 1, 0)
    result_map[:, :, :, 9,:] = np.where(vegetation, 1, 0)
    result_map[:, :, :, 10,:] = np.where(ground, 1, 0)
    result_map[:, :, :, 11,:] = np.where(sky, 1, 0)
    result_map[:, :, :, 12,:] = np.where(person, 1, 0)
    result_map[:, :, :, 13,:] = np.where(rider, 1, 0)
    result_map[:, :, :, 14,:] = np.where(car, 1, 0)
    result_map[:, :, :, 15,:] = np.where(truck, 1, 0)
    result_map[:, :, :, 16,:] = np.where(bus, 1, 0)
    result_map[:, :, :, 17,:] = np.where(motorcycle, 1, 0)
    result_map[:, :, :, 18,:] = np.where(bicycle, 1, 0)

    return result_map


# Data generator for fit_generator.


def data_generator_cnn(b_size):

    f1 = np.load('deep_feature_2.npy')  # 599 256 512 19
    deep_feature = np.squeeze(f1)            # 256 256 512  1 19
    deep_feature = deep_feature[:,:,:,np.newaxis,:].transpose(0,1,2,4,3)
    # f2 = np.load('shallow_feature.npy')  #
    f2=np.load('flow_1.npy')         # 256 256 512 2 19
    # shallow_feature = np.squeeze(f2)
    flow = np.squeeze(f2).transpose(0,1,2,4,3)
    label = np.load('label_1.npy')  # 256 256 512

    x_imgs1 = deep_feature[:deep_feature.shape[0]-1,:,:,:,:]#original_data
    # x_imgs2 = shallow_feature[1:,:,:,:]
    x_imgs2= flow[:flow.shape[0]-1:,:,:,:]        # 255
    y_imgs = label[1:,:,:]        # 255

    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args()
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = x_imgs1.shape[0]  #256
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]  #0....254随机数
            # c=x_imgs1[idx]
            # cc=x_imgs2[idx]
            x.append(np.concatenate([x_imgs1[idx],x_imgs2[idx]],axis= 3))
            y.append(y_imgs[idx].reshape(256,512,1,1))

            if len(x) == b_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(b_size).reshape(b_size,1,1,1)
                seed = random.randrange(1, 100)

                # x_tmp_gen = x_data_gen.flow(np.array(x), _,
                #                             batch_size=b_size,
                #                             seed=seed)

                # y_tmp_gen = y_data_gen.flow(np.array(y), _,
                #                             batch_size=b_size,
                #                             seed=seed)

                # Finally, yield x, y data.
                # x_result, _ = next(np.array(x))
                # y_result, _ = next(np.array(y))

                yield np.array(x), get_result_map(b_size,np.array(y))

                x.clear()
                y.clear()

