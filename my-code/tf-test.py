import tensorflow as tf
import  numpy as np
import os
import argparse
import h5py
import cv2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

vc = cv2.VideoCapture('./data_video/video00.mp4')
print(vc.isOpened())

size=(int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)))
#视频编码
# fourcc=int(vc.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'avc1')
#视频帧
fps = vc.get(cv2.CAP_PROP_FPS)
vw_name = './data_video/'+'result_video00'+'.mp4'
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
i=0
f = h5py.File('data.h5', 'w')
output_graph_path='./weight/FCN0_model_weight.pb'
with open(output_graph_path, "rb") as f:
        output_graph_def = tf.GraphDef()
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            input_x = sess.graph.get_tensor_by_name("input_1:0")
            output = sess.graph.get_tensor_by_name("activation_18/truediv:0")

            while i < 15:
                    ok, imgInput = vc.read()
                    if ok is False:  # the end of the tesed video
                        break

                    imgInput = cv2.resize(imgInput, (512, 256), interpolation=cv2.INTER_AREA)
                    b, g, r = cv2.split(imgInput)
                    img_rgb = cv2.merge([r, g, b])
                    input_data = img_rgb / 127.5 - 1

                    input_data = np.expand_dims(input_data, 0)
                    t1 = time.time()
                    res = sess.run(output, {input_x: input_data})
                    print("predict time = %d" % ((time.time() - t1) * 1000))
                    res = result_map_to_img(res[0])  # 10ms
                    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                    # cv2.imshow('res', res)
                    mask = cv2.addWeighted(imgInput, 0.5, res, 0.6, 0.0)

                    #
                    # cv2.imshow('show', imgShow)
                    #
                    # # cv2.waitKey(24)
                    vw.write(mask)
                    # key_cv = cv2.waitKey(1)
                    # if key_cv == 27:
                    #     break
                    i += 1





