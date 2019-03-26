from __future__ import print_function

import argparse
import cv2
import numpy as np

from PIL import Image
from model.unet import unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from model.ASPP import ASPP
from model.aspp_shallow import fcn_shallow
from skimage import io,transform
from model.one_layer import simple_layer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

with open('./define/colors.txt') as color:
    colors=color.read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in colors]
    COLORS = np.array(COLORS, dtype="uint8")

with open('./define/classes.txt') as f:
    labels=f
def result_map_to_img(res_map):
    t_start = cv2.getTickCount()
    img = np.zeros((256, 512, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)  #去除冗余维度（256,512,19）
    # img=np.zeros((256,512,3),type=np.uint8)
    argmax_idx = np.argmax(res_map, axis=2)   #(mask-map) （256,512）
    #mask

    for label_id, label in enumerate(COLORS):
        # c=(argmax_idx==label_id)
        img[argmax_idx==label_id]= COLORS[label_id]

    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print("label Time: %.3f ms" % t_total)

    return img


# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'ASPP', 'pspnet','unet'],
                    help="Model to test. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-P", "--img_path", required=True, help="The image path you want to test")
parser.add_argument("-c", "--classes", required=True, default='--classes define/classes.txt', help="path to .txt file containing class labels")
parser.add_argument("-l", "--colors", required=True,type=str, default='--colors define/colors.txt', help="path to .txt file containing colors for labels")

args = parser.parse_args()
model_name = 'fcn'
img_path = args.img_path
c=vars(args)
CLASSES = open(c["classes"]).read().strip().split("\n")

# Use only 3 classes.
# labels = ['background', 'person', 'car', 'road']
t_start = cv2.getTickCount()
# Choose model to test

model_shallow = fcn_shallow(input_shape=(256, 512, 3))
model_1_1= simple_layer(input_shape=(64, 128, 128), num_classes=len(CLASSES))

try:
    model_shallow.load_weights('./weight/'+model_name + '_model_weight.h5', by_name=True)

except:
    print("You must train model and get weight before test.")

try:
    model_1_1.load_weights('./weight/' + model_name + '_model_weight.h5', by_name=True)
except:
    print("mdoel1_1 must train and get weight before test")

t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
print("load Time: %.3f ms" % t_total)

x_img = cv2.imread(img_path)
x_img=cv2.resize(x_img, (512,256), interpolation=cv2.INTER_AREA)
origin=x_img
cv2.imshow('origin_pic', origin)
x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

x_img = x_img / 127.5 - 1
x_img = np.expand_dims(x_img, 0)

t_start = cv2.getTickCount()
pred1 = model_shallow.predict(x_img)                                                 # 2000ms
pred2=model_1_1.predict(pred1[0])



t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
print("Predict Time: %.3f ms" % t_total)

res = result_map_to_img(pred2[0])   # 10ms

cv2.imshow('res', res)

origin=cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
mask = cv2.addWeighted(origin, 0.5, res, 0.6, 0.0)
cv2.imshow('mask', mask)

cv2.waitKey(0)



