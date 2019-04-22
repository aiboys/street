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
        # print(sess2.run(K.sum(y_true * y_pred)))
        # print(sess2.run(K.sum(y_true*y_true)))
        # print(sess2.run( K.sum(y_pred*y_pred)))
        print(sess2.run(K.square(y_true-y_pred)))
        print(sess2.run(K.sum(K.square(y_true-y_pred))))
    # return K.mean(2.*(K.sum(y_true * y_pred)) / (K.sum(y_true*y_true) + K.sum(y_pred*y_pred)))    ##用这个函数去判别区分度
    return K.sqrt(K.sum(K.square(y_true - y_pred)))
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


#
#
# c=np.array([ [[1,0.1,0.8],[1,0,0],[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1]], [[1,0,1],[1,0,0],[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1]] ])
# cc=np.array([ [[1,0,1],[1,0,0],[1,1,1]], [[1,0,1],[1,0,1],[1,1,1]] ])
# c=np.array([ [1,1,1], [1,3,1]           ])
# cc=np.array([  [1,1,1], [1,1,0]                   ])
# # c=c.flatten()
# # cc=cc.flatten()
# c=tf.convert_to_tensor(c)
# cc=tf.convert_to_tensor(cc)
# c=K.cast(c,dtype=tf.float32)
# cc=K.cast(cc,dtype=tf.float32)
#
# ccc=dice_coef(c,cc)
#
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(ccc))
#
#
#
# print(c.shape)
#
#
# import numpy as np
# import time
# from numba import vectorize
#
# # @vectorize(["float32(float32, float32)"], target='cuda')
# # def vectorAdd(a, b):
# #     return a + b
# #
# #
# # N = 320000000
# #
# # A = np.ones(N, dtype=np.float32 )
# # B = np.ones(N, dtype=np.float32 )
# # C = np.zeros(N, dtype=np.float32 )
# #
# # start = time.time()
# # C = vectorAdd(A, B)
# # vectorAdd_time = (time.time() - start)*1000
# #
# # # print("c[:5] = " + str(C[:5]))
# # # print("c[-5:] = " + str(C[-5:]))
# #
# # print("vectorAdd took %f seconds with GPU" % vectorAdd_time)
# #
# #
# # import numpy as np
# # import time
# #
# # def vectorAdd(a, b):
# #     return a + b
# #
# # N = 320000000
# #
# # A = np.ones(N, dtype=np.float32 )
# # B = np.ones(N, dtype=np.float32 )
# # C = np.zeros(N, dtype=np.float32 )
# #
# #
# # start = time.time()
# # C = vectorAdd(A, B)
# # vectorAdd_time = (time.time() - start)*1000
# # print("vectorAdd took %f seconds with CPU " % vectorAdd_time)
#
#
#
# import numpy as np
# import keras.backend as K
# import tensorflow as tf
#
# t1 = K.variable(np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]]))
# t2 = K.variable(np.array([[[7, 4], [8, 4]], [[2, 10], [15, 11]]]))
#
# d0 = K.concatenate([t1, t2], axis=0)
# d1 = K.concatenate([t1, t2], axis=1)
# d2 = K.concatenate([t1, t2], axis=2)
# d3 = K.concatenate([t1, t2], axis=-1)
# cc=[]
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     # print(sess.run(d0.get_shape().as_list()))
#     print(d0.get_shape().as_list())
#     for i in range(3):
#         c=d0.eval()
#         cc.append(c)
#
#     np.save("c.npy",cc)
#     f=np.load("c.npy")
#     ccc=f[0]
#     print(ccc)
# import tensorflow as tf
# import numpy as np
# c=np.arange(2490368).reshape((256,512,19,1))
# ccc=c
# cc=[]
# # for i in range(2):
# cc.append(np.concatenate([c,c],axis=3))
# cc.append(np.concatenate([ccc,c],axis=3))
#
#
# c1=np.array(cc)
# for i in range(3):
#    cc=c
#    cc=np.concatenate([cc,c],axis=3)
# # ccc=c.shape[3]
#
# pass

import numpy as np


# print(2/3)
a=np.arange(4).reshape((2,2)).astype('float64')
#
# # print(a[:,:])
# aaaa=np.ones_like(a)
# c=a.sum(axis=1)
# b=np.array([2,3])
# print(b/5)
# for i in range(2):
#     print(a[i,:])
#     print(c[i])
#     cc=a[i,:]/c[i]
#     aaaa[i,:]=cc
# print(aaaa)
# pass
# # c=aaa+a
# # ccc=c.sum(0)
# aa=(a<4) & (a>0) & (a!=1)
# # print(a[aa])
# # c=a[aa]
# a=a.flatten()
# aaa=aaa.flatten()
#
# print(np.bincount(2*a[aa].astype+ aaa[aa], minlength= 4).reshape(2,2))

from sklearn import metrics
import xlwt as excel
from decimal import Decimal
import json

def plot_confusion_matrix(label_num,hist, labels_name, title):
    # plt.figure(figsize=(8, 2), dpi=80)
    ind_array = np.arange(label_num)
    tick_marks = np.array(range(label_num)) + 0.5
    x, y = np.meshgrid(ind_array, ind_array)
    hist = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]    # 归一化
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = hist[y_val][x_val]
        if c >=0:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.imshow(hist, interpolation='nearest')
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def label_mapping(input, mapping):  # 主要是因为CityScapes标签里面原类别太多，这样做把其他类别转换成算法需要的类别（共19类）和背景（标注为255）
    output = np.copy(input)  # 先复制一下输入图像
    raw,clo=input.shape
    flag=0
    for i in range(raw):
        for j in range(clo):
            c=input[i,j]
            if input[i,j] not in mapping[:,0] :
                flag=1
                print(input[i,j])
                input[i,j]=0
    print(flag)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]  # 进行类别映射，最终得到的标签里面之后0-18这19个数加255（背景）

    return np.array(output, dtype=np.int64)  # 返回映射的标签

def per_class_iu(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))



def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别

    return np.bincount(n*a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,n)

a=np.array([[7,8,8,11],[7,8,22,11]])
aa=np.array([[7,7,7,7],[7,8, 22, 0]])

cc=[1,2,3,4,5]
print(cc.reverse())

# for i,j in enumerate(cc):
#     if j<3:
#         ii=i
# k=['b','c']
cc.insert(4,'a')
ccc=[1,3,5]
# with open('info.json', 'r') as fp:
#     info = json.load(fp)
# num_classes = np.int(info['classes'])  # 读取类别数目，这里是19类，详见博客中附加的info.json文件
# print('max Num classes', num_classes)  # 打印一下类别数目
# name_classes = np.array(info['label'], dtype=np.str)  # 读取类别名称，详见博客中附加的info.json文件
# mapping = np.array(info['label2train'], dtype=np.int)  # 读取标签映射方式，详见博客中附加的info.json文件
#
# a=label_mapping(a,mapping)
# aa=label_mapping(aa,mapping)
# hist=metrics.confusion_matrix(a.flatten(),aa.flatten())
# # hist = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]
# hist[np.isnan(hist)]=0
#
# miou=per_class_iu(hist)
# print(miou)
# plot_confusion_matrix(5,hist,cc, "HAR Confusion Matrix")
# # plt.savefig('/HAR_cm.png', format='png')
# plt.show()
# pass




# c=np.array([[1,2,3],[22,9,10]])
# cc=np.bincount(c.flatten(), minlength=34)
#
# np.savetxt('c.txt',cc)
# a=np.array([[0,1,255,2],[0,1,255,2]])
# aa=np.array([[0,0,0,0],[0,0,255,2]])
labels_name=['background',
'road',
'sidewalk',
'building',
'wall',
'fence',
'pole',
'trafficLight',
'trafficSign',
'vegetation',
'terrain',
'sky',
'person',
'rider',
'car',
'truck',
'bus',
'motorcycle',
'bicycle']

dic_class={}




with open('info.json', 'r') as fp:
    info = json.load(fp)
num_classes = np.int(info['classes'])  # 读取类别数目，这里是19类，详见博客中附加的info.json文件
print('max Num classes', num_classes)  # 打印一下类别数目
name_classes = np.array(info['label'], dtype=np.str)  # 读取类别名称，详见博客中附加的info.json文件
mapping = np.array(info['label2train'], dtype=np.int)  # 读取标签映射方式，详见博客中附加的info.json文件
data = h5py.File('./dataset_parser/data.h5', 'r')
val_path = data.get('/test' + '/y')  # 注意这里的数据经过flatten 只有200张图片~~~~~   200
size = val_path.shape[0]

for idx in range(2):
    label_img= val_path[idx].reshape(256,512)
    label_img = label_mapping(label_img, mapping=mapping)
    lab=np.bincount(label_img.flatten(),minlength=19)   #最多19类
    print(sum(lab))
    np.savetxt('./batch_test_result/label_class_num/'+'lab_'+str(idx)+'_.txt', lab)
    lab_record=[]
    pre_record=[]
    class_num=0
    for i in range(len(lab)):
        if lab[i] !=0 :
            lab_record.append(i)   #记录有哪些类别
            class_num+=1  # 记录总类别数
    print("class_num:\t {:d}".format(class_num))
    labels_name_new=[]
    for i in lab_record:
        labels_name_new.append(labels_name[i])  #保存别类名
    if idx == 0:
        hist_dict={}

    pre=cv2.imread('./batch_test_result/'+str(idx)+'_label.png',cv2.IMREAD_GRAYSCALE)
    pre_lab = np.bincount(pre.flatten(), minlength=19)
    for i in range(len(pre_lab)):
       if pre_lab[i] !=0:
           if i not in lab_record:  # 出现新的类别编号
                print(i)
                pre_record.append(i)
                for v,k in enumerate(lab_record):
                    if k<i:
                       print(v)
                    else:
                       labels_name_new.insert(v,labels_name[i])   # 如果出现新的编号，则插入label_name
                       break

    # pre=label_mapping(pre,mapping=mapping)

    # file = excel.Workbook(encoding='utf-8')
    # table1 = file.add_sheet('hist')
    # raw,clo=label_img.shape

    # file.save('label.xls')

    # a=label_mapping(a,mapping)
    # aa=label_mapping(aa,mapping)
    # hist=fast_hist(a.flatten(),aa.flatten(),4)
    # lab=np.bincount(label_img.flatten(),minlength=19)
    # print(sum(lab))
    # np.savetxt('label_imgsadsada321321s.txt',lab)
    hist = metrics.confusion_matrix(label_img.reshape(-1), pre.reshape(-1))
    # hist_for_show = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]
    #
    # label_name_for _show=[]
    # plot_confusion_matrix(label_num= len(labels_name_new), hist=hist, labels_name=labels_name_new, title="HAR Confusion Matrix")
    # plt.savefig('/HAR_cm.png', format='png')
    # plt.show()



    raw,_= hist.shape
    file = excel.Workbook(encoding='utf-8')
    table1 = file.add_sheet('hist')
    hist_csv = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]
    hist_csv[np.isnan(hist_csv)] = 0
    for i in range(raw):
        for j in range(raw):
               table1.write(i, j, Decimal(float(hist_csv[i, j])).quantize(Decimal("0.000")))
    file.save('./batch_test_result/label_class_num/'+'hist_'+str(idx)+'_.xls')

    miou = per_class_iu(hist=hist)
    raw,_=hist.shape
    for i in range(raw):
        if idx == 0:
           hist_dict[str(labels_name_new[i])]= 0        # 新建键值对，值全是0
        elif labels_name_new[i] not in hist_dict.keys():  # 更新：
           hist_dict[str(labels_name_new[i])] = 0       # 如果新的hist类别不在是新的，则创建，初始值为0
        hist_dict[str(labels_name_new[i])]+=miou[i]     # 将miou写入字典中 并进行累加

for key, value in hist_dict.items():
    print('class======>{:s}\nmiou======>{:2f}\n'.format(key,value/(idx+1)*100))




raw,_=hist.shape
np.savetxt('hista_aa.txt',hist)
hist = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]
miou=per_class_iu(hist=hist)
print(miou)
plot_confusion_matrix(hist, labels_name, "HAR Confusion Matrix")
# plt.savefig('/HAR_cm.png', format='png')
plt.show()

pass