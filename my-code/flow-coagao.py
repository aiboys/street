# import numpy as np
# import cv2
#
#
# cap = cv2.VideoCapture( 'J:\\video\\img_to_video\\img_to_video_01_1.mp4')
#
# #获取第一帧
# ret, frame1 = cap.read()
# frame1=cv2.resize(frame1, (512, 256), interpolation=cv2.INTER_AREA)
# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#
#
#
# hsv = np.zeros_like(frame1)    #1080 1920 3
#
# #遍历每一行的第1列
# hsv[...,1] = 255
#
#
#
# while(1):
#  ret, frame2 = cap.read()
#  if ret:
#     next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#     next=cv2.resize(next, (512, 256), interpolation=cv2.INTER_AREA)
#     # t_start = cv2.getTickCount()
#     #返回一个两通道的光流向量，实际上是每个点的像素位移值
#     flow = cv2.calcOpticalFlowFarneback(prvs,prvs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     a=flow[...,0]
#     b=flow[...,1]
#     #print(flow.shape)
#     # print(flow)
#     print(flow[...,0])
#     print(flow[...,1])
#     #笛卡尔坐标转换为极坐标，获得极轴和极角
#     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#     rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#     # t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000     #每一帧都需要 39ms    256 512 3
#     # print(t_total)
#     # cv2.imshow('frame2',rgb)
#     # k = cv2.waitKey(30) & 0xff
#     # if k == 27:
#     #     break
#     # elif k == ord('s'):
#     #     cv2.imwrite('opticalfb.png',frame2)
#     #     cv2.imwrite('opticalhsv.png',rgb)
#     prvs = next
#  else:
#      break
#
# cap.release()
# cv2.destroyAllWindows()









import numpy as np
import cv2
import math
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def bi_linear(src, dst, target_size):
    pic = cv2.imread(src)       # 读取输入图像
    pic = cv2.resize(pic, (512, 256),interpolation=cv2.INTER_AREA)
    th, tw = target_size[0], target_size[1]
    emptyImage = np.zeros(target_size, np.uint8)
    t_start = cv2.getTickCount()
    for i in range(th):
            for j in range(tw):
                # 首先找到在原图中对应的点的(X, Y)坐标
                corr_x = (i+0.5)/th*pic.shape[0]-0.5
                corr_y = (j+0.5)/tw*pic.shape[1]-0.5
                # if i*pic.shape[0]%th==0 and j*pic.shape[1]%tw==0:     # 对应的点正好是一个像素点，直接拷贝
                #   emptyImage[i, j, k] = pic[int(corr_x), int(corr_y), k]
                point1 = (math.floor(corr_x), math.floor(corr_y))   # 左上角的点
                point2 = (point1[0], point1[1]+1)
                point3 = (point1[0]+1, point1[1])
                point4 = (point1[0]+1, point1[1]+1)

                fr1 = (point2[1]-corr_y)*pic[point1[0], point1[1],:] + (corr_y-point1[1])*pic[point2[0], point2[1],:]
                fr2 = (point2[1]-corr_y)*pic[point3[0], point3[1],:] + (corr_y-point1[1])*pic[point4[0], point4[1],:]
                emptyImage[i, j, :] = (point3[0]-corr_x)*fr1 + (corr_x-point1[0])*fr2


    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print(t_total)
    # emptyImage=cv2.resize(emptyImage, (512, 256), interpolation=cv2.INTER_AREA)
    # return emptyImage



src = 'K:\\1.png'
dst = 'K:\\new_1.png'
target_size = (255, 510, 3)     # 变换后的图像大小

bi_linear(src, dst, target_size)
# cv2.imwrite(dst, m)


