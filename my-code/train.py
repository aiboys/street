from __future__ import print_function
import csv
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from callbacks import TrainCheck

from model.unet import unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from model.ASPP import ASPP
from dataset_parser.generator import data_generator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet','ASPP'],
                    help="Model to train. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-TB", "--train_batch", required=False, default=16, help="Batch size for train.")
parser.add_argument("-VB", "--val_batch", required=False, default=8, help="Batch size for validation.")
parser.add_argument("-LI", "--lr_init", required=False, default=3e-4, help="Initial learning rate.")
parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")
parser.add_argument("-c", "--classes", required=True, default='--classes define/classes.txt', help="path to .txt file containing class labels")
parser.add_argument("-l", "--colors", required=True,type=str, default='--colors define/colors.txt', help="path to .txt file containing colors for labels")

args = parser.parse_args()
print(args)
model_name = args.model
TRAIN_BATCH = args.train_batch
VAL_BATCH = args.val_batch
lr_init = args.lr_init
lr_decay = args.lr_decay
vgg_path = args.vgg
c=vars(args)
CLASSES = open(c["classes"]).read().strip().split("\n")

print(CLASSES)

# Use only 3 classes.
# labels = ['background', 'person', 'car', 'road']

# if a colors file was supplied, load it from disk
if c["colors"]:
	COLORS = open(c["colors"]).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")
	print ('All COLORS: \n{0}'.format(COLORS))

# otherwise, we need to randomly generate RGB colors for each class
# label
else:
	# initialize a list of colors to represent each class label in the mask (starting with 'black' for the background/unlabeled regions)
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
		dtype="uint8")
	COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

# initialize the legend visualization
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")


# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(CLASSES),
                   lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(256, 512, 3), num_classes=len(CLASSES),
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=lr_init, lr_decay=lr_decay)
elif model_name=="ASPP":
    model = ASPP(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=lr_init, lr_decay=lr_decay)
# Define callbacks

checkpoint = ModelCheckpoint(filepath='./weight/'+model_name + '_model_weight.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)

# train_check = TrainCheck(output_path='./img', model_name=model_name)
#early_stopping = EarlyStopping(monitor='val_dice_coef', patience=10)

# training
# aa,bb=data_generator('dataset_parser/data.h5', TRAIN_BATCH, 'train')
history = model.fit_generator(data_generator('dataset_parser/data.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch=512 // TRAIN_BATCH,
                              validation_data=data_generator('dataset_parser/data.h5', VAL_BATCH, 'val'),
                              validation_steps=256// VAL_BATCH,
                              callbacks=[checkpoint],
                              epochs=150,
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


history_loss=open('./model-train-result/histor_loss.csv','a',newline='')
csv_write1=csv.writer(history_loss,dialect='excel')
loss=history.history['loss']
csv_write1.writerow(loss)
history_loss.close()

history_val_loss=open('./model-train-result/histor_val_loss.csv','a',newline='')
csv_write2=csv.writer(history_val_loss,dialect='excel')
val_loss=history.history['val_loss']
csv_write2.writerow(val_loss)
history_val_loss.close()

history_dice_coef=open('./model-train-result/histor_dice_coef.csv','a',newline='')
csv_write3=csv.writer(history_dice_coef,dialect='excel')
dice_coef=history.history['dice_coef']
csv_write3.writerow(dice_coef)
history_dice_coef.close()

history_val_dice_coef=open('./model-train-result/histor_val_dice_coef.csv','a',newline='')
csv_write4=csv.writer(history_val_dice_coef,dialect='excel')
val_dice_coef=history.history['val_dice_coef']
csv_write4.writerow(val_dice_coef)
history_val_dice_coef.close()
















