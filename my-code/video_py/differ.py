import cv2
import numpy as np
import tensorflow as tf

from keras import backend as K

def difference(key_frame,current_frame):

    pic1 = cv2.cvtColor(key_frame, cv2.COLOR_RGB2GRAY)
    pic2=cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    pic1=pic1/127.5-1
    pic2=pic2/127.5-1

    # r,g,b=key_frame[:,:,0],key_frame[:,:,1],key_frame[:,:,2]
    # r2,g2,b2=current_frame[:,:,0],current_frame[:,:,1],current_frame[:,:,2]
    # pic1=r*0.229+g*0.587+b*0.114
    # pic2=r2*0.229+g2*0.587+b2*0.114

    pic1= cv2.resize(pic1, (8,8), interpolation=cv2.INTER_CUBIC)
    pic2=cv2.resize(pic2, (8,8), interpolation=cv2.INTER_CUBIC)
    d=tf.cast((pic1-pic2),dtype=tf.float32)
    differ=tf.sqrt(tf.cast(tf.reduce_sum(tf.square(d)), dtype=tf.float32))

    return differ

#
# def dice_coef(y_true, y_pred):
#     y_true=y_true.flatten()
#     y_pred=y_pred.flatten()
#     y_true=tf.convert_to_tensor(y_true)
#     y_pred=tf.convert_to_tensor(y_pred)
#     y_true=K.cast(y_true, dtype=tf.float32)
#     y_pred=K.cast(y_pred, dtype=tf.float32)
#     y_pred=y_pred/255
#     y_true=y_true/255
#
#     return (K.mean(2.*(K.sum(y_true * y_pred))) / (K.sum(y_true*y_true) + K.sum(y_pred*y_pred)))
#     # return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)    #这个应该是正式版的，但是对于图片的相似度没有任何用啊？？？why??
#
# # # def draw_difference(differ):
#
# pic1=cv2.imread('J:\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00\stuttgart_00_000000_000001_leftImg8bit.png')
# pic2=cv2.imread('J:\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00\stuttgart_00_000000_000001_leftImg8bit.png')
#
# pic1=cv2.cvtColor(pic1,cv2.COLOR_RGB2GRAY)
# pic2=cv2.cvtColor(pic2,cv2.COLOR_RGB2GRAY)
#
# c=dice_coef(pic1, pic2)
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(c))

# def draw(difference):
import os
x_paths=[]
theta=1.05
import matplotlib.pyplot as plt
def draw_difference():

        tmp_img_folder_path='J:\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00'
        for (path, dirname, files) in os.walk(tmp_img_folder_path):
            for filename in files:
                x_paths.append(os.path.join(path, filename))
        i=0
        key=[]
        diff_record=[]
        flag=0
        while True:
            img_path_1 = x_paths[i]
            pic=cv2.imread(img_path_1)
            pic=cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
            if i==0:
               key.append(pic)
               flag=1
            diff = difference(key_frame=key[len(key) - 1], current_frame=pic)
            sess = tf.Session()
            diff_record.append(diff.eval(session=sess))

            if i!=0:
                if diff.eval(session=sess)>theta:
                    key.append(pic)
                    flag=1
                else:
                    flag=0
            i=i+1
            if flag==1:
                 plt.plot(i, diff_record[len(diff_record)-1], 'ro')
                 plt.annotate("(%s,%s)" % (i,diff_record[len(diff_record)-1]), xy=(i,diff_record[len(diff_record)-1]), xytext=(-20, 10), textcoords='offset points')
            else:
                 plt.plot(i, diff_record[len(diff_record) - 1], 'bo')

            if i==60:
                break
        plt.plot([theta]*i,'r--')
        plt.show()

def diff_plot(diff_record,flag_record,frame,theta_record):  #diff存储的是区别值，key_frame存储的是关键帧位置，

    for i in range(frame):
          if flag_record[i]==1:
              plt.plot(i, diff_record[i], 'ro')
              plt.annotate("(%s)" % i, xy=(i, diff_record[len(diff_record)-1]), xytext=(-20, 10), textcoords='offset points')
          else:
              plt.plot(i,diff_record[i],'bo')

    plt.plot([theta_record]*frame, 'r--')
    plt.show()




