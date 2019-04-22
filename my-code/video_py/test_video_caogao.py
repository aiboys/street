from __future__ import print_function
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras
import argparse
import cv2
import numpy as np
from keras.models import Model
from model.one_layer import simple_layer
from model.fcn_shallow import fcn_shallow
from model.fcn_deep import fcn_deep
from video_py.differ import difference,diff_plot
import os
import time
from model.with_cnn_deep  import with_cnn_deep
from model.with_cnn_shallow import with_cnn_shallow
from model.CNN import CNN
from model.FCN_0 import FCN



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
    img = np.zeros((512,1024, 3), dtype=np.uint8)
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
    if flag==1:
       print("Predict key frame shallow total time: %.3f ms" % t_total)
    else:
        print("Predict current frame shallow total time: %.3f ms" % t_total)
    return  result_shallow

def deep_predict(pic,flag):
    input=pic
    t_start = cv2.getTickCount()
    result = model_deep.predict(input)
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    if flag==1:
        print("Predict key frame deep total time: %.3f ms" % t_total)
    else:
        print("Predict current frame deep total time: %.3f ms" % t_total)
    return result

def mask(pic):
    input=pic
    imgmask = result_map_to_img(input[0])
    return imgmask


# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'FCN0', 'pspnet','fcn2'],
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


model_fcn = FCN(input_shape=(512,1024,3), num_classes= len(CLASSES), lr_init=3e-4,lr_decay=5e-4)
#fcn2:
model_shallow = with_cnn_shallow(input_shape=(512, 1024, 3))
#deep layer：
model_deep= with_cnn_deep(input_shape=(512,1024,19),num_classes=len(CLASSES))
#（1,1）layer -the end：
# model_CNN= CNN(input_shape=(256,512,19),num_classes=len(CLASSES),lr_init=3e-4,lr_decay=5e-4)

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')


# vc = cv2.VideoCapture('./data-video/'+'video'+video_num+'.mp4')
vc = cv2.VideoCapture('../data_video/video00.mp4')
print(vc.isOpened())

size=(int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)))
#视频编码
# fourcc=int(vc.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'avc1')
#视频帧
fps = vc.get(cv2.CAP_PROP_FPS)
vw_name = '../data_video/'+'result_video_WITH_CNN'+video_num+'.mp4'
#输出视频
vw = cv2.VideoWriter(vw_name,fourcc,25,(512,256))
# # vw.open(vw_name, fourcc, fps, size)


config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


shallow=Model(inputs=model_fcn.input,outputs=model_fcn.get_layer('act19').output)
deep= Model(inputs=model_deep.input, outputs=model_deep.output)
try:
    model_shallow.load_weights('../weight/'+model_name+'_model_weight.h5',by_name=True)
except:
    print("dsadsadsadasdas")
try:
    model_deep.load_weights("../weight/"+model_name+'_model_weight.h5',by_name=True)
except:
     print("dsadasdasdsadasdasdasdasdasdas2132131")

#load models:
t_start = cv2.getTickCount()
print("load wights start...")
try:
    model_fcn.load_weights('../weight/'+model_name + '_model_weight.h5', by_name=True)

except:
    print("model_fcn must be trained before test.")
#
try:
    shallow.load_weights('../weight/'+model_name+'_model_weight.h5', by_name=True)
except:
    print("shallow must be trained before test")
try:
    deep.load_weights('../weight/'+model_name+'_model_weight.h5', by_name=True)
except:
    print("deep must be trained before test")
# try:
#     model_deep.load_weights('../weight/' + model_name + '_model_weight.h5', by_name=True)
# except:
#     print("model_deep must be trained before test")
#
# try:
#     model_fcn.load_weights("../weight/"+model_name+'_model_weight.h5',by_name=True)
# except:
#     print("fcn must be ....")

t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
print("[*] model loading Time: %.3f ms\n" % t_total)


frame=0  #记录关键帧
key_frame_record=[]
key_feature=[]
key_frame=[]
theta=1.8
diff=0
diff_record=[]
flag=0
flag_record=[]
i=0
# video segmentation:
while i<1:

    ok, imgInput = vc.read()
    if ok is False:   #the end of the tesed video
        break

    imgInput = cv2.resize(imgInput, (1024, 512), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(imgInput)
    img_rgb = cv2.merge([r, g, b])
    input_data=img_rgb/127.5-1
    input_data = np.expand_dims(input_data, 0)

    # 整体一次性跑完
    res1 = model_fcn.predict(input_data)
    res1 = mask(res1)
    cv2.imwrite(str(i)+'_1_.png', res1)


    # 整个中间+后层：
    shallow_2=shallow.predict(input_data)
    # result_2=model_deep.predict(shallow_2)
    mask2 = mask(shallow_2)
    cv2.imwrite(str(i)+'_2_.png',mask2)


    # 重构的浅层+深层：
    shallow_3= model_shallow.predict(input_data)
    result_3= model_deep.predict(shallow_3)
    mask3= mask(result_3)
    cv2.imwrite(str(i)+'_3_.png',mask3)
    #
    if (shallow_2-result_3).all() == (np.zeros_like(shallow_2)).all():
         print("相等")





    i+=1

vc.release()
vw.release()

# file=open('KEY_FRAME.txt','w')
# file.write(str(key_frame_record))
# file.close()
# key_c=cv2.waitKey()

# diff_plot(diff_record=diff_record,flag_record=flag_record,frame=frame,theta_record=theta)










