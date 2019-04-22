import keras.backend as K
from keras.layers.core import Activation, Dropout, Lambda
import numpy as np
import random
import tensorflow as tf
import cv2
import sys
sys.path.append('..')
import os
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from video_py.readnpy_ds2l import data_generator_cnn

# from video_py.readnpy_ds2l_with_val import data_generator_cnn_with_val
from model.CNN import CNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#GPU '0' is available:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name="CNN"
model=CNN(num_classes=19,input_shape=(256,512,19),lr_init=3e-4,lr_decay=5e-4)

checkpoint = ModelCheckpoint(filepath='../weight/' + model_name + '_model_weight.h5',
                                 monitor='val_dice_coef',
                                 save_best_only=False,
                                 save_weights_only=True)

# train_check = TrainCheck(output_path='./img', model_name=model_name)
# early_stopping = EarlyStopping(monitor='val_dice_coef', patience=10)

# training

history = model.fit_generator(data_generator_cnn( b_size=1),
                                  steps_per_epoch=5 // 1,
                                  # validation_data= data_generator_cnn_with_val(b_size=1),
                                  # validation_steps= 2 / 1,
                                  callbacks=[checkpoint],
                                  epochs=1,
                                  verbose=1)




history_loss=open('../model_train_result/cnn_histor_loss.csv','a',newline='')
csv_write1=csv.writer(history_loss,dialect='excel')
loss=history.history['loss']
csv_write1.writerow(loss)
history_loss.close()

history_val_loss=open('../model_train_result/cnn_histor_val_loss.csv','a',newline='')
csv_write2=csv.writer(history_val_loss,dialect='excel')
val_loss=history.history['val_loss']
csv_write2.writerow(val_loss)
history_val_loss.close()

history_dice_coef=open('../model_train_result/cnn_histor_dice_coef.csv','a',newline='')
csv_write3=csv.writer(history_dice_coef,dialect='excel')
dice_coef=history.history['dice_coef']
csv_write3.writerow(dice_coef)
history_dice_coef.close()

history_val_dice_coef=open('../model_train_result/cnn_histor_val_dice_coef.csv','a',newline='')
csv_write4=csv.writer(history_val_dice_coef,dialect='excel')
val_dice_coef=history.history['val_dice_coef']
csv_write4.writerow(val_dice_coef)
history_val_dice_coef.close()