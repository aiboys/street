import tensorflow as tf
import  numpy as np
import os
import argparse
import h5py
import cv2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# with open('../define/colors.txt') as color:
#     colors=color.read().strip().split("\n")
#     COLORS = [np.array(c.split(",")).astype("int") for c in colors]
#     COLORS = np.array(COLORS, dtype="uint8")
#
# with open('../define/classes.txt') as f:
#     labels=f

def convert(input):
    imgInput = cv2.resize(input, (512, 256), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(imgInput)
    img_rgb = cv2.merge([r, g, b])
    input_data = img_rgb / 127.5 - 1
    return input_data

def flow(prvs,next):
    ASPPflow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    x= ASPPflow[..., 0]
    y= ASPPflow[..., 1]
    flow_result = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]], axis=2)
    return flow_result

def result_map_to_label(res_map):
    res_map = np.squeeze(res_map)  #去除冗余维度（256,512,21）
    argmax_idx = np.argmax(res_map, axis=2)   #(mask-map) （256,512）
    argmax_idx=argmax_idx.astype(np.int16)
    return argmax_idx
# Parse Options
parser = argparse.ArgumentParser()
# parser.add_argument("-P", "--img_path", required=True, help="The image path you want to test")
# parser.add_argument("-c", "--classes", required=True, default='--classes define/classes.txt', help="path to .txt file containing class labels")
# parser.add_argument("-l", "--colors", required=True,type=str, default='--colors define/colors.txt', help="path to .txt file containing colors for labels")

args = parser.parse_args()
# model_name = args.model
# img_path = args.img_path
c=vars(args)

vc = cv2.VideoCapture('../data_video/video00.mp4')
print(vc.isOpened())

size=(int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)))
#视频编码
# fourcc=int(vc.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'avc1')
#视频帧
fps = vc.get(cv2.CAP_PROP_FPS)
vw_name = '../data_video/'+'result_video01'+'.mp4'
#输出视频
vw = cv2.VideoWriter(vw_name,fourcc,25,(512,256))
# # vw.open(vw_name, fourcc, fps, size)
#FCN:
# input is : input:0
# output is: act/truediv:0
#ASPP:
# input is : input_1:0
# output is: activation_9/truediv:0
#ASPP2:
# input is : input_1:0
# output is: activation_11/truediv:0
# FCN0:
# input is : input_1:0
# output is: activation_18/truediv:0
# CNN:
# input is : input_1:0
# output is: activation_6/truediv:0
i=0
output_graph_path='../weight/FCN0_model_weight.pb'
deep_feature=[]
shallow_feature=[]
label_result=[]
# shallow_flow=[]
shallow_feature_flow=[]
# with open(graph2, 'rb') as f2:
#     grapf2_def = tf.GraphDef()
#     grapf2_def.ParseFromString(f2.read())
#     _ = tf.import_graph_def(grapf2_def, name="")

with open(output_graph_path, "rb") as f:
        output_graph_def = tf.GraphDef()
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.initialize_all_variables().run()


            # constant_ops = [op for op in sess.graph.get_operations()]
            # for name in constant_ops:
            #     print(name.name)

            input_x = sess.graph.get_tensor_by_name("input_1:0")
            shallow=sess.graph.get_tensor_by_name("lambda_1/resize_images/ResizeBilinear:0")
            output = sess.graph.get_tensor_by_name("activation_19/truediv:0")
            # shallow = sess.graph.get_tensor_by_name("activation_7/Relu:0")
            i=0
            # ok0, imgInput0 = vc.read()
            while i<6:
                ok, imgInput1 = vc.read()
                if ok is False:  # the end of the tesed video
                        break
                    # input_data0=convert(imgInput0)
                if i % 3 == 0:
                    input_data1 = convert(imgInput1)
                    # input_data0 = np.expand_dims(input_data0, 0)
                    input_data1=np.expand_dims(input_data1,0)
                    # t1 = time.time()
                    # deep feature:
                    deep = np.squeeze(sess.run(output, {input_x: input_data1}))
                    #第一帧shallow feature：
                    # shallow_0=np.squeeze(sess.run(shallow,{input_x:input_data0}))
                    # 第二帧shallow feature：
                    shallow_1=np.squeeze(sess.run(shallow,{input_x: input_data1}))
                    # 第一帧label：
                    label = result_map_to_label(deep)  # 10ms







            ########  保存npy文件：

                    deep_feature.append(deep)
                    shallow_feature.append(shallow_1)
                    label_result.append(label)
                    print(i)
                i+=1
            #
            #
            # f1=np.save('deep_feature_3.npy',deep_feature)
            # # f2=np.save('shallow_feature_1.npy',shallow_feature)
            # f3=np.save('label_3.npy',label_result)
            # f4=np.save('shallow_feature_3.npy',shallow_feature)

            f = h5py.File('feature5.h5', 'w')
            f.create_dataset('deep_feature_5', data= deep_feature)
            f.create_dataset('shallow_feature_5', data = shallow_feature)
            f.create_dataset('label_5', data = label_result)
            print("over...")
            f.close()



# f = h5py.File('feature3.h5', 'r')
#
# cc=f['deep_feature_3'][:]
# pass