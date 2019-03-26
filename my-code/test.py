from __future__ import print_function
import argparse
import cv2
import numpy as np
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from model.ASPP import ASPP
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
    res_map = np.squeeze(res_map)  #去除冗余维度（256,512,21）
    # print(res_map[:,:,1])
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
model_name = args.model
img_path = args.img_path
c=vars(args)
CLASSES = open(c["classes"]).read().strip().split("\n")

# Use only 3 classes.
# labels = ['background', 'person', 'car', 'road']
t_start = cv2.getTickCount()
# Choose model to test
if model_name == "fcn":
    model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)
elif model_name == "ASPP":
    model = ASPP(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)

try:
    model.load_weights('./weight/'+model_name + '_model_weight.h5', by_name=True)
except:
    print("You must train model and get weight before test.")



t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
print("load Time: %.3f ms" % t_total)

x_img = cv2.imread(img_path)
# x_img=x_img.resize((256,512),Image.ANTIALIAS)
x_img=cv2.resize(x_img, (512,256), interpolation=cv2.INTER_AREA)
origin=x_img
# cv2.imshow('origin_pic', origin)
cv2.imwrite('./output-img/origin-img.png',x_img)
x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

x_img = x_img / 127.5 - 1
x_img = np.expand_dims(x_img, 0)

t_start = cv2.getTickCount()
pred = model.predict(x_img)                                                 # 2000ms
t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
print("Predict Time: %.3f ms" % t_total)

res = result_map_to_img(pred[0])   # 10ms
res= cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
# cv2.imshow('res', res)

cv2.imwrite('./output-img/res.png',res)
# origin=cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
mask = cv2.addWeighted(origin, 0.5, res, 0.6, 0.0)
# cv2.imshow('mask', mask)
cv2.imwrite('./output-img/mask.png',mask)
# cv2.waitKey(0)
