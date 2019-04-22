from __future__ import print_function
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import tensorflow as tf
import argparse
import cv2
import numpy as np
from model.fcn import fcn_8s
from model.ASPP import ASPP
from model.aspp2 import ASPP2
from model.FCN_0 import FCN
import os
import time
from model.resnet50 import resnet

import json
import h5py
import matplotlib.pyplot as plt
import xlwt as excel
from decimal import Decimal
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))


#GPU '0' is available:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#GPU for different parts:
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



with open('./define/colors.txt') as color:
    colors=color.read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in colors]
    COLORS = np.array(COLORS, dtype="uint8")
    color.close()

with open('./define/classes.txt', 'r') as f:
    CLASSES = f.read().strip().split("\n")

def result_map_to_label(res_map):

    img = np.zeros((512, 1024, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)  #去除冗余维度（256,512,21）

    argmax_idx = np.argmax(res_map, axis=2)   #(mask-map) （256,512）
    argmax_idx=argmax_idx.astype(np.int16)

    return argmax_idx

def result_map_to_img(res_map):

    t_begin=time.time()
    img = np.zeros((512, 1024, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)  #去除冗余维度（256,512,21）

    argmax_idx = np.argmax(res_map, axis=2)   #(mask-map) （256,512）
    argmax_idx=argmax_idx.astype(np.int16)


    #mask

    for label_id, label in enumerate(COLORS):
        # c=(argmax_idx==label_id)
        img[argmax_idx==label_id]= COLORS[label_id]
    t_end=time.time()
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print("label Time: %.3f ms" % t_total)
    print("time:%s"%((t_end-t_begin)*1000))
    return img

# def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
#     '''
# 	核心代码
# 	'''
#     k = (a >= 0) & (a <= n) & (a !=16)   # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别
#     return np.bincount(n*a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,n)
#
#     # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)

def per_class_iu(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)


def label_mapping(input, mapping):  # 这样做把其他类别转换成算法需要的类别（共18类）和背景（标注为255）
    output = np.copy(input)  # 先复制一下输入图像
    raw,clo=input.shape
    for i in range(raw):
        for j in range(clo):
            if input[i,j] not in mapping[:,0]:
                input[i,j]=0

    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]  # 进行类别映射，最终得到的标签里面之后0-18这19个数加255（背景）

    return np.array(output, dtype=np.int64)  # 返回映射的标签

def plot_confusion_matrix(label_num,hist, labels_name, title):
    plt.figure(figsize=(8, 4), dpi=100)
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



def test_batch(model, h5_path,test_num, outdir=''):
    labels_name = ['background',
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
    with open( 'info.json', 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])  # 读取类别数目，这里是18类
    print('Num classes', num_classes)  # 打印一下类别数目
    name_classes = np.array(info['label'], dtype=np.str)  # 读取类别名称
    mapping = np.array(info['label2train'], dtype=np.int)  # 读取标签映射方式
    hist = np.zeros((num_classes, num_classes))  # hist初始化为全零，在这里的hist的形状是[19, 19]
    hist_sum=[]
    # 存储多张图片的hist矩阵
    arg=[]
    data = h5py.File(h5_path,'r')
    val_path = data.get('/test'+'/y')   #注意这里的数据经过flatten 只有200张图片~~~~~   200
    size= val_path.shape[0]
    test_path= data.get('/test'+'/x')
    f = h5py.File(os.path.join(outdir,'label_test.h5'),'w')

    #  label images:
    for idx in range(test_num):
         test_img = test_path[idx].reshape((256,512,3))
         test_img= cv2.resize(test_img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
         x_img = test_img / 127.5 - 1
         x_img = np.expand_dims(x_img, 0)

         res = model.predict(x_img)

         res_map = np.squeeze(res)  # 去除冗余维度（256,512,21）
         argmax= np.argmax(res_map, axis=2)  # (mask-map) （256,512）
         cv2.imwrite(os.path.join(outdir, str(idx)+'_label.png'), argmax)
         arg.append(argmax)
         print('label {:d}'.format(idx))


    #  caculate moiu
    f.create_dataset('label_test', data=arg)
    f = h5py.File(os.path.join(outdir,'label_test.h5'),'r')
    test_img = f['label_test'][:]
    for idx in range(test_num):
        test_img_= test_img[idx]
        test_img_ = np.array(test_img_)
        label = np.array(val_path[idx].reshape(256, 512))
        label=cv2.resize(label,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

        label_img = label_mapping(label,mapping)
        lab = np.bincount(label_img.flatten(), minlength=19)  # 最多19类
        print(sum(lab))
        np.savetxt('./batch_test_result/label_class_num/' + 'lab_' + str(idx) + '_.txt', lab)
        lab_record = []
        pre_record = []
        class_num = 0
        if len(label.flatten()) != len(test_img_.flatten()):
            print('Skipping: len(label) = {:d}, len(test) = {:d}, num= {:s}'.format(len(label.flatten()),
                                                                                     len(test_img_.flatten()),
                                                                                     str(idx)))

            continue

        for i in range(len(lab)):
            if lab[i] != 0:
                lab_record.append(i)  # 记录有哪些类别
                class_num += 1  # 记录总类别数
        print("class_num:\t {:d}".format(class_num))
        labels_name_new = []
        for i in lab_record:
            labels_name_new.append(labels_name[i])  # 保存别类名
        if idx == 0:
            hist_dict = {}

        pre = cv2.imread('./batch_test_result/' + str(idx) + '_label.png', cv2.IMREAD_GRAYSCALE)
        pre_lab = np.bincount(pre.flatten(), minlength=19)
        for i in range(len(pre_lab)):
            if pre_lab[i] != 0:
                if i not in lab_record:  # 出现新的类别编号
                    print(i)
                    pre_record.append(i)
                    for v, k in enumerate(lab_record):
                        if k < i:
                            pass
                             #print(v)
                        else:
                            labels_name_new.insert(v, labels_name[i])  # 如果出现新的编号，则插入label_name
                            break

        hist = metrics.confusion_matrix(label_img.reshape(-1), pre.reshape(-1))
        # hist_for_show = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]

        # plot_confusion_matrix(label_num=len(labels_name_new), hist=hist, labels_name=labels_name_new,title="HAR Confusion Matrix")
        # plt.savefig('./batch_test_result/'+str(idx)+'_HAR_hist.png', format='png')
        # plt.show()
        # plt.close()

        raw, _ = hist.shape
        file = excel.Workbook(encoding='utf-8')
        table1 = file.add_sheet('hist')
        hist_csv = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]
        hist_csv[np.isnan(hist_csv)] = 0
        for i in range(raw):
            for j in range(raw):
                table1.write(i, j, Decimal(float(hist_csv[i, j])).quantize(Decimal("0.000")))
        file.save('./batch_test_result/label_class_num/' + 'hist_' + str(idx) + '_.xls')

        miou = per_class_iu(hist=hist)
        raw, _ = hist.shape
        for i in range(raw):
            if idx == 0:
                hist_dict[str(labels_name_new[i])] = 0  # 新建键值对，值全是0
            elif labels_name_new[i] not in hist_dict.keys():  # 更新：
                hist_dict[str(labels_name_new[i])] = 0  # 如果新的hist类别不在是新的，则创建，初始值为0
            hist_dict[str(labels_name_new[i])] += miou[i]  # 将miou写入字典中 并进行累加
    hist_dict_for_show = hist_dict

    a=0
    for key, value in hist_dict.items():
            print('class======>{:s}\nmiou======>{:2f}\n'.format(key, value / (idx + 1) * 100))
            a=a+value/(idx+1)*100


    for key , value in hist_dict_for_show.items():
        hist_dict_for_show[key]=value/(idx+1)*100
    hist_dict_for_show["miou"] = a / len(hist_dict)
    json_str = json.dumps(hist_dict_for_show, indent=4)
    with open('./batch_test_result/miou.json', 'w') as json_file:
                json_file.write(json_str)

        # hist = metrics.confusion_matrix(label.flatten(), test_img_.flatten()) # 计算每一张图片的混淆矩阵
        # # hist = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]   # 归一化
        # print('{:d}/{:d}th total image_mIoU is: {:0.3f}'.format(idx, size, np.mean(per_class_iu(hist)) * 100))
        # hist_sum.append(hist)

    # plot hist

    # mIoUs = per_class_iu(hist_sum)
    # for id_class in range(len(num_classes)):
    #     print("====>" + name_classes[id_class] + ':\t' + str(round(mIoUs[id_class]*100, 2)))
    # print('=======> mIoU: ' + str(round(np.nanmean(mIoUs)*100,2)))
    fp.close()
    f.close()




def test_simple_img(model, img_path, h5_path=None):
    x_img = cv2.imread(img_path)
    x_img = cv2.resize(x_img, (1024, 512), interpolation=cv2.INTER_AREA)
    origin = x_img
    # cv2.imshow('origin_pic', origin)

    output_path = './output_img/'
    img_path_new= os.path.basename(img_path)

    cv2.imwrite(output_path+'origin_'+img_path_new, x_img)
    x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

    x_img = x_img / 127.5 - 1
    x_img = np.expand_dims(x_img, 0)

    t_start = cv2.getTickCount()
    pred = model.predict(x_img)
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print("Predict Time: %.3f ms" % t_total)

    res = result_map_to_img(pred[0])  # 10ms
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # cv2.imshow('res', res)

    cv2.imwrite(output_path+'res_'+img_path_new, res)
    mask = cv2.addWeighted(origin, 0.5, res, 0.6, 0.0)
    # cv2.imshow('mask', mask)
    cv2.imwrite(output_path+'mask_'+img_path_new, mask)
    # cv2.waitKey(0)
    print('test image {:s} over...'.format(img_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--model", required=True, choices=['fcn', 'ASPP','resnet','ASPP2','FCN0'],
                        help="Model to test. 'fcn', 'unet', 'pspnet' is available.")
    parser.add_argument("-P", "--img_path", required=True, help="The image path you want to test")
    parser.add_argument("-T", "--test_flag", required=True, choices=['0','1'],
                        help = "choose test model. '0' goes to test single image, '1' goes to test bacth images.")
    # parser.add_argument("-c", "--classes", required=True, default='--classes define/classes.txt', help="path to .txt file containing class labels")
    # parser.add_argument("-l", "--colors", required=True,type=str, default='--colors define/colors.txt', help="path to .txt file containing colors for labels")

    args = parser.parse_args()
    model_name = args.model
    img_path = args.img_path
    test_flag = args.test_flag


    t_start = cv2.getTickCount()
    # Choose model to test
    if model_name == "fcn":
      model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)
    elif model_name == "resnet" :
       model = resnet(input_shape= (256,512,3),num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)
    elif model_name == "FCN0" :
       model = FCN(input_shape= (512,1024,3),num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)
    elif model_name == "ASPP":
        model = ASPP(input_shape= (256,512,3),num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)
    elif model_name == "ASPP2":
        model = ASPP2(input_shape=(256, 512, 3), num_classes=len(CLASSES), lr_init=3e-4, lr_decay=5e-4)

    try:
        model.load_weights('./weight/'+model_name + '_model_weight.h5',by_name=True)
    except:
        print("You must train model and get weight before test.")
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print('load model:  {:s}  Time: {:3f} ms'.format(model_name, t_total))

    h5_path='./dataset_parser/data.h5'
    outdir= './batch_test_result'
    if test_flag == '0':  #test single image
        test_simple_img(model=model, img_path=img_path)

    elif test_flag == '1':
        test_batch(model=model,h5_path=h5_path, outdir= outdir, test_num=2)

















































































































































