import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import h5py
# a='./dataset/'+'data.h5'
# c=h5py.File(a, 'w')
import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-N", "--num", required=True, choices=['00','01','02'], help="The video num you want to test : 00,01,02...")
#
# args = parser.parse_args()
# c=args.num
# print(c)
# parser = argparse.ArgumentParser()
# parser.add_argument("--path", required=True, default=".\\dataset\\leftImg8bit",help="path of leftImg8bit folder.")  #--path path
# args = parser.parse_args()
# img_folder_path = args.path
# img_folder_path="./dataset/leftImg8bit/train"
# y=len(img_folder_path)
# x_paths=[]
# x=0
# for (path, dirname, files) in os.walk(img_folder_path):
#     for filename in files:
#         x_paths.append(os.path.join(path, filename))
#         x=x+1
#         if x>5:
#             break

# with open('./define/classes.txt','r') as f:
#     c=f.read().strip().split("\n")
#     print(c)
import numpy as np
# with open('./define/colors.txt', 'r') as f:
#     C = f.read().strip().split("\n")
#     COLORS = [np.array(c.split(",")).astype("int") for c in C]
#     COLORS = np.array(COLORS, dtype="uint8")
#     print(COLORS[2][2])
# import cv2
# import cv2
# print(cv2.__version__)


from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

# a=K.variable(np.array([1,2,3]))
# b=K.variable(np.array([4,5,6]))
# c=K.variable(np.array([7,8,9]))
# d=[]
# dd=[]
# ddd=K.variable(np.array([7,8,9]))
# dddd=K.variable(np.array([7,8,9]))
# ddddd=K.variable(np.array([7,8,9]))
# dd.append(ddd)
# dd.append(dddd)
# dd.append(ddddd)
# d.append(a)
# d.append(b)
# d.append(c)
# init = tf.global_variables_initializer()
# e=concatenate(d,dd)
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(e))


#
# c= tf.constant([[1,2],[1,1]])
# cc = tf.constant([[2,2],[2,2]])
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(K.sum(c*cc)))
#
#
# # k=[]
# # k.append(c)
# # k.append(cc)
# # c=len(k)
# init=tf.global_variables_initializer()
# # ccc=tf.square(c-cc)
# ccc= tf.sqrt(tf.cast(tf.reduce_sum(tf.square(c-cc)),dtype=tf.float32))
# with tf.Session() as sess:
#      sess.run(init)
#      print(sess.run(ccc))
# ccc=c+cc

# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(c))
#     print(sess.run(cc))
#     print(sess.run(ccc))


# x=[]
# c=1
# cc=2
# ccc=3
# x.append(c)
# x.append(cc)
# x.append(ccc)
# for i in x:
#      print(i)


import cv2


def dice_coef(y_true, y_pred):



    init2 = tf.global_variables_initializer()
    with tf.Session() as sess2:
        sess2.run(init2)
        print(sess2.run(K.sum(y_true * y_pred)))
        print(sess2.run(K.sum(y_true*y_true)))
        print(sess2.run( K.sum(y_pred*y_pred)))

    return K.mean(2.*(K.sum(y_true * y_pred)) / (K.sum(y_true*y_true) + K.sum(y_pred*y_pred)))    ##用这个函数去判别区分度
    # return K.mean(2.*(K.sum(y_true * y_pred)) / (K.sum(y_true*y_true) + K.sum(y_pred*y_pred)))    ##用这个函数去判别区分度

    # return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)   #right one  smooth=1,加快收敛     只能是0~1 的测试图和 0,1的标签


# c= K.cast((tf.constant([[1,2],[1,1]])),dtype=tf.float32)
# c=c.flatten()
# cc =K.cast( (tf.constant([[2,2],[2,2]])),dtype=tf.float32)
# cc=cc.flatten()
# im=cv2.imread('J:\leftImg8bit_demoVideo\\fine\\stuttgart_00_000000_000001_leftImg8bit_json\\label.png')
# y_img = cv2.resize(im[:,:,0], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
# y_img=tf.convert_to_tensor(y_img)
# y_img=K.cast(y_img,dtype=tf.float32)
# # cv2.imshow('dsa',y_img)
# im2=cv2.imread('J:\leftImg8bit_demoVideo\\fine\\stuttgart_00_000000_000001_leftImg8bit_json\\label.png')
# y_img2=cv2.resize(im2[:,:,0],None,fx=0.25,fy=0.25,interpolation=cv2.INTER_NEAREST)
# # cv2.simshow()
# y_img2=tf.convert_to_tensor(y_img2)
# y_img2=K.cast(y_img2,dtype=tf.float32)
# k=dice_coef(c,cc)




c=np.array([ [[1,0.1,0.8],[1,0,0],[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1]], [[1,0,1],[1,0,0],[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1]] ])
# cc=np.array([ [[1,0,1],[1,0,0],[1,1,1]], [[1,0,1],[1,0,1],[1,1,1]] ])
# c=c.flatten()
# cc=cc.flatten()
cc=c
c=tf.convert_to_tensor(c)
cc=tf.convert_to_tensor(cc)
c=K.cast(c,dtype=tf.float32)
cc=K.cast(cc,dtype=tf.float32)

ccc=dice_coef(c,cc)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(ccc))



print(c.shape)


import numpy as np
import time
from numba import vectorize

@vectorize(["float32(float32, float32)"], target='cuda')
def vectorAdd(a, b):
    return a + b


N = 320000000

A = np.ones(N, dtype=np.float32 )
B = np.ones(N, dtype=np.float32 )
C = np.zeros(N, dtype=np.float32 )

start = time.time()
C = vectorAdd(A, B)
vectorAdd_time = (time.time() - start)*1000

# print("c[:5] = " + str(C[:5]))
# print("c[-5:] = " + str(C[-5:]))

print("vectorAdd took %f seconds with GPU" % vectorAdd_time)


import numpy as np
import time

def vectorAdd(a, b):
    return a + b

N = 320000000

A = np.ones(N, dtype=np.float32 )
B = np.ones(N, dtype=np.float32 )
C = np.zeros(N, dtype=np.float32 )


start = time.time()
C = vectorAdd(A, B)
vectorAdd_time = (time.time() - start)*1000
print("vectorAdd took %f seconds with CPU " % vectorAdd_time)
