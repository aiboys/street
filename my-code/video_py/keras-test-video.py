from __future__ import print_function
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras
import argparse
import cv2
import numpy as np
from model.FCN_0 import FCN
from model.one_layer import simple_layer
from model.fcn_shallow import fcn_shallow
from model.fcn_deep import fcn_deep
from video_py.differ import difference,diff_plot
import os
import time
from model.fcn2_deep import fcn2_deep
from model.fcn2_shallow import fcn2_shallow
from model.simple2_layer import simple2_layer
# GPU  '0' is available：
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#GPU for different parts:
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

with open('../define/colors.txt') as color:
    colors=color.read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in colors]
    COLORS = np.array(COLORS, dtype="uint8")


with open('../define/classes.txt','r') as f:
    CLASSES= f.read().strip().split("\n")

def result_map_to_img(res_map):
    img = np.zeros((256, 512, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)  # 去除冗余维度（256,512,19）

    argmax_idx = np.argmax(res_map, axis=2)   #(mask-map) （256,512）

    for label_id,label in enumerate(COLORS):
        img[argmax_idx==label_id]= COLORS[label_id]

    return img


def shallow_predict(pic,flag):
    input=pic
    t_start = cv2.getTickCount()
    result_shallow = model_shallow.predict(input)  # (1,64,128,128)
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    # if flag==1:
    #    print("Predict key frame shallow total time: %.3f ms" % t_total)
    # else:
    #     print("Predict current frame shallow total time: %.3f ms" % t_total)
    return  result_shallow

def deep_predict(pic,flag):
    input=pic
    t_start = cv2.getTickCount()
    result = model_deep.predict(input)
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    # if flag==1:
    #     print("Predict key frame deep total time: %.3f ms" % t_total)
    # else:
    #     print("Predict current frame deep total time: %.3f ms" % t_total)
    return result

def one_predict(pic):
    input=pic
    t_start = cv2.getTickCount()
    result_deep = model_1_1.predict(input)
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    # if flag==1:
    #     print("Predict current frame 1X1 total time: %.3f ms" % t_total)
    return result_deep

def mask(pic):
    input=pic
    t_start = cv2.getTickCount()
    imgmask = result_map_to_img(input[0])
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    # print("mask frame time: %.3f ms" % t_total)
    return imgmask


# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet','fcn2','FCN0'],
                    help="Model to test. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-N", "--num", required=True, choices=['00','01','02'], help="The video num you want to test : 00,01,02...")
# parser.add_argument("-c", "--classes", required=True, default='--classes define/classes.txt', help="path to .txt file containing class labels")
# parser.add_argument("-l", "--colors", required=True,type=str, default='--colors define/colors.txt', help="path to .txt file containing colors for labels")

args = parser.parse_args()
model_name = args.model
video_num=args.num



# shallow layer： fcn:
# model_shallow = fcn_shallow(input_shape=(256, 512, 3))
# #deep layer：
# model_deep= fcn_deep(input_shape=(64,128,128),num_classes=len(CLASSES))
# #（1,1）layer -the end：
# model_1_1= simple_layer(input_shape=(64,128,128),num_classes=len(CLASSES))



#fcn2:
model_shallow = fcn2_shallow(input_shape=(256, 512, 3))
#deep layer：
model_deep= fcn2_deep(input_shape=(64,128,128),num_classes=len(CLASSES))
#（1,1）layer -the end：
model_1_1= simple2_layer(input_shape=(64,128,128),num_classes=len(CLASSES))

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')


print("load model_weights...\n")
# vc = cv2.VideoCapture('./data-video/'+'video'+video_num+'.mp4')
vc = cv2.VideoCapture('../data_video/video00.mp4')
print(vc.isOpened())

size=(int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)))
#视频编码
# fourcc=int(vc.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'avc1')
#视频帧
fps = vc.get(cv2.CAP_PROP_FPS)
vw_name = '../data_video/'+'result_video'+video_num+'.mp4'
#输出视频
vw = cv2.VideoWriter(vw_name,fourcc,25,(512,256))
# # vw.open(vw_name, fourcc, fps, size)


config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

if model_name== "FCN0":
   model = FCN(input_shape= (256,512,3),num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)

try:
   model.load_weights('./weight/'+model_name + '_model_weight.h5',by_name=True)
except:
    print("You must train model and get weight before test.")




frame=0  #记录关键帧
key_frame_record=[]
key_feature=[]
key_frame=[]
theta=1.8
diff=0
diff_record=[]
flag=0
flag_record=[]
# video segmentation:
for i in range(10):
    ok, imgInput = vc.read()
    if ok is False:   #the end of the tesed video
        break
    imgInput = cv2.resize(imgInput, (512, 256), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(imgInput)
    img_rgb = cv2.merge([r, g, b])
    input_data=img_rgb/127.5-1

    input_data = np.expand_dims(input_data, 0)
    res = model.predict(input_data)
    imgmask=result_map_to_img(res[0])
    imgShow = cv2.cvtColor(imgInput, cv2.COLOR_RGB2BGR)
#    imgMaskColor = imgmask
    imgShow = cv2.addWeighted(imgShow, 0.5, imgmask, 0.6, 0.0)
    #
    # cv2.imshow('show', imgShow)
    #
    # # cv2.waitKey(24)
    vw.write(imgShow)
    # key_cv = cv2.waitKey(1)
    # if key_cv == 27:
    #     break


vc.release()
vw.release()

# file=open('KEY_FRAME.txt','w')
# file.write(str(key_frame_record))
# file.close()
# key_c=cv2.waitKey()

# diff_plot(diff_record=diff_record,flag_record=flag_record,frame=frame,theta_record=theta)

