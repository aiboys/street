from __future__ import print_function
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import csv
import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from callbacks import TrainCheck
from keras.models import load_model
from model.fcn import fcn_8s
from model.fcn2 import fcn_8s_2
from model.resnet50 import resnet
from model.ASPP import ASPP
from model.aspp2 import  ASPP2
from dataset_parser.generator_video import data_generator_video
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

#GPU '0' is available:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#GPU for different parts:
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'fcn2','ASPP','resnet','ASPP2'],
                    help="Model to train. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-TB", "--train_batch", required=False, default=1, help="Batch size for train.")
parser.add_argument("-VB", "--val_batch", required=False, default=1, help="Batch size for validation.")
parser.add_argument("-LI", "--lr_init", required=False, default=3e-4, help="Initial learning rate.")
parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")
# parser.add_argument("-c", "--classes", required=False, default='--classes ./define/classes2.txt', help="path to .txt file containing class labels")
# parser.add_argument("-l", "--colors", required=False,type=str, default='--colors ./define/colors2.txt', help="path to .txt file containing colors for labels")

args = parser.parse_args()
model_name = args.model
TRAIN_BATCH = args.train_batch
VAL_BATCH = args.val_batch
lr_init = args.lr_init
lr_decay = args.lr_decay
vgg_path = args.vgg
# CLASSES = open(c["classes"]).read().strip().split("\n")

with open('../define/classes2.txt','r') as f:
    CLASSES= f.read().strip().split("\n")



# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(CLASSES),
                   lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "fcn2":
    model = fcn_8s_2(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=lr_init, lr_decay=lr_decay)
elif model_name=="ASPP":
    model = ASPP(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=lr_init, lr_decay=lr_decay)
# Define callbacks
elif model_name == "ASPP2":
    model = ASPP2(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=lr_init, lr_decay=lr_decay)
elif model_name=="resnet":
    model = resnet(input_shape=(256,512,3),num_classes=len(CLASSES),lr_init=lr_init,lr_decay=lr_decay)


checkpoint = ModelCheckpoint(filepath='../weight/'+model_name + '_model_weight.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)


# train_check = TrainCheck(output_path='./img', model_name=model_name)
#early_stopping = EarlyStopping(monitor='val_dice_coef', patience=10)

# training

history = model.fit_generator(data_generator_video('../dataset_parser/data.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch=1 // TRAIN_BATCH,
                              validation_data=data_generator_video('../dataset_parser/data.h5', VAL_BATCH, 'val'),
                              validation_steps=1// VAL_BATCH,
                              callbacks=[checkpoint],
                              epochs=1,
                               verbose=1)


# plt.title("loss")
# plt.plot(history.history["loss"], color="r", label="train")
# plt.plot(history.history["val_loss"], color="b", label="val")
# plt.legend(loc="best")
# plt.savefig('model-train-result/'+model_name + '_loss.png')
#
# plt.gcf().clear()
# plt.title("dice_coef")
# plt.plot(history.history["dice_coef"], color="r", label="train")
# plt.plot(history.history["val_dice_coef"], color="b", label="val")
# plt.legend(loc="best")
# plt.savefig('model-train-result/'+model_name + '_dice_coef.png')


history_loss=open('../model_train_result_video/histor_loss.csv','a',newline='')
csv_write1=csv.writer(history_loss,dialect='excel')
loss=history.history['loss']
csv_write1.writerow(loss)
history_loss.close()

history_val_loss=open('../model_train_result_video/histor_val_loss.csv','a',newline='')
csv_write2=csv.writer(history_val_loss,dialect='excel')
val_loss=history.history['val_loss']
csv_write2.writerow(val_loss)
history_val_loss.close()

history_dice_coef=open('../model_train_result_video/histor_dice_coef.csv','a',newline='')
csv_write3=csv.writer(history_dice_coef,dialect='excel')
dice_coef=history.history['dice_coef']
csv_write3.writerow(dice_coef)
history_dice_coef.close()

history_val_dice_coef=open('../model_train_result_video/histor_val_dice_coef.csv','a',newline='')
csv_write4=csv.writer(history_val_dice_coef,dialect='excel')
val_dice_coef=history.history['val_dice_coef']
csv_write4.writerow(val_dice_coef)
history_val_dice_coef.close()
















