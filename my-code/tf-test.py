import tensorflow as tf
import  numpy as np
import os
import argparse

import cv2
import time
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

    argmax_idx = np.argmax(res_map, axis=2)   #(mask-map) （256,512）
    argmax_idx=argmax_idx.astype(np.int16)
    np.savetxt("filename.txt", argmax_idx)

    #mask

    for label_id, label in enumerate(COLORS):
        # c=(argmax_idx==label_id)
        img[argmax_idx==label_id]= COLORS[label_id]

    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print("label Time: %.3f ms" % t_total)

    return img


# Parse Options
parser = argparse.ArgumentParser()
# parser.add_argument("-P", "--img_path", required=True, help="The image path you want to test")
parser.add_argument("-c", "--classes", required=True, default='--classes define/classes.txt', help="path to .txt file containing class labels")
parser.add_argument("-l", "--colors", required=True,type=str, default='--colors define/colors.txt', help="path to .txt file containing colors for labels")

args = parser.parse_args()
# model_name = args.model
# img_path = args.img_path
c=vars(args)
#FCN:
# input is : input:0
# output is: act/truediv:0
#ASPP:
# input is : input_1:0
# output is: activation_9/truediv:0
#ASPP2:
# input is : input_1:0
# output is: activation_11/truediv:0
output_graph_path='./weight/ASPP_model_weight.pb'
with open(output_graph_path, "rb") as f:
        output_graph_def = tf.GraphDef()
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            input_x = sess.graph.get_tensor_by_name("input_1:0")
            output = sess.graph.get_tensor_by_name("activation_9/truediv:0")
            in_data ='./dataset/test.png'

            x_img = cv2.imread(in_data)
            # x_img=x_img.resize((256,512),Image.ANTIALIAS)
            x_img = cv2.resize(x_img, (512, 256), interpolation=cv2.INTER_AREA)
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
            origin=x_img
            x_img = x_img / 127.5 - 1
            x_img = np.expand_dims(x_img, 0)

            t1=time.time()
            y_conv_2 = sess.run(output, {input_x: x_img})
            t2=time.time()
            print('predict time:%s'%((t2-t1)*1000))

            res = result_map_to_img(y_conv_2[0])  # 10ms
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./output-img/res.png', res)
            # origin=cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
            mask = cv2.addWeighted(origin, 0.5, res, 0.6, 0.0)
            # cv2.imshow('mask', mask)
            cv2.imwrite('./output-img/mask.png', mask)





