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
# hsv = np.zeros_like(frame1)    #1080 1920 3
#
# #遍历每一行的第1列
# hsv[...,1] = 255
#
# while(1):
#  ret, frame2 = cap.read()
#  if ret:
#     next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#     next=cv2.resize(next, (512, 256), interpolation=cv2.INTER_AREA)
#     t_start = cv2.getTickCount()
#     #返回一个两通道的光流向量，实际上是每个点的像素位移值
#     flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000     #每一帧都需要 39ms    256 512 3
#     print(t_total)
#     a=flow[...,0]
#     b=flow[...,1]
#     #print(flow.shape)
#     # print(flow)
#     # print(flow[...,0])
#     # print(flow[...,1])
#     #笛卡尔坐标转换为极坐标，获得极轴和极角
#     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#     # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#
#     cv2.imshow('frame2',hsv)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv2.imwrite('opticalfb.png',frame2)
#         cv2.imwrite('opticalhsv.png',hsv)
#     prvs = next
#  else:
#      break
#
# cap.release()
# cv2.destroyAllWindows()


#
# import numpy as np
# import cv2
# import time
# import math
# import matplotlib.pyplot as plt
#
#
# def flow(prvs,next):
#     flow_io = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     a= flow_io[...,0]
#     b= flow_io[..., 1]
#     return a, b
#
#
#
#
# c=np.arange(3200).reshape((40,80,1))
# c2=np.arange(10,3210).reshape((40,80,1))
# hsv1=np.concatenate([c,c2],axis=2)
# hsv1=np.concatenate([hsv1,c],axis=2)
# hsv=np.zeros_like(hsv1)
#
#
# hsv[...,1]=255
#
#
# a, b= flow(c,c2)
# # cc=np.concatenate([a[:,:,np.newaxis],b[:,:,np.newaxis]],axis=2)
# mag, ang = cv2.cartToPolar(a, b)
# hsv[..., 0] = ang * 180 / np.pi / 2
# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
# # cv2.imshow('da',hsv)
# # rgb = hsv2rgb(h=hsv[...,0], s=hsv[...,1], v=hsv[...,2])
#
# plt.imshow(rgb)
# # cv2.waitKey(0)
# plt.show()
# pass

#
# def warp(prvs_path,next_path):
#     prvs=cv2.imread(prvs_path)
#     next=cv2.imread(next_path)
#     prvs1=cv2.resize(prvs, (512,256), interpolation=cv2.INTER_AREA)
#     prvs = cv2.cvtColor(prvs1, cv2.COLOR_BGR2GRAY)
#     next=cv2.resize(next, (512,256), interpolation=cv2.INTER_AREA)
#     next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
#
#     hsv=np.zeros_like(prvs1)
#     result_img=prvs1
#     result_img2=result_img
#     hsv[..., 1] = 255
#     a, b = flow(prvs=prvs, next=next)
#
#     mag, ang = cv2.cartToPolar(a, b)
#     hsv[..., 0] = ang*180/np.pi/2
#     hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#     rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#     cv2.imwrite('output-img/flow.png', rgb)
#
#     t = time.time()
#     for i in range(256):
#         for j in range(512):
#
#               # ii= math.floor(i-a[i,j])
#               # jj = math.floor(j-b[i,j])
#               # point1 = (math.floor(ii), math.floor(jj))  # 左上角的点
#               # point3 = (point1[0], point1[1] + 1)  # 左下角
#               # point2 = (point1[0] + 1, point1[1])  # 右上角
#               # point4 = (point1[0] + 1, point1[1] + 1)  # 右下角
#               #
#               # if (point1[0]>255) or (point1[0]<0) or (point2[0]>255) or (point2[0]<0) or (point3[0]>255) or( point3[0]<0) or (point4[0]>255 )or (point4[0]<0) or (point1[1]>511) or (point1[1]<0) or (point2[1]>511)  or (point2[1]<0) or( point3[1]>511) or( point3[1]<0 )or (point4[1]>511) or (point4[1]<0):
#               #     result_img2[i,j,:]=result_img[i,j,:]
#               #
#               # else:
#               #    # r2=math.floor((point3[1]-jj)/(point3[1]-point1[1])) *result_img[point1[0],point1[1],:]+ math.floor((jj-point1[1])/(point3[1]-point1[1]))*result_img[point3[0],point3[1],:]
#               #    # r1=math.floor((point4[1]-jj)/(point4[1]-point2[1]))*result_img[point2[0],point2[1],:]+ math.floor((jj-point2[1])/(point4[1]-point2[1]))*result_img[point4[0],point4
#               #    r2 =(point3[1] - jj) * result_img[point1[0], point1[1], :] + (jj - point1[1]) * result_img[point3[0], point3[1], :]
#               #    r1 = (point4[1] - jj) * result_img[point2[0], point2[1], :] + (jj - point2[1]) * result_img[point4[0], point4[1], :]
#               #    result_img2[i,j,:]=(point1[0]-ii)*r1 + (ii-point4[0])*r2
#
#               # if ii<0 or ii>255 or jj<0 or jj>511:
#               # result_img2[i,j,:]=result_img[i,j,:]
#               # else:
#               #     result_img2[i,j,:]=result_img[ii,jj,:]
#                pass
#
#
#     print('t=: %d'%((time.time()-t)*1000))
#     return result_img2


#
# prvs_path='J:\\video1.png'
# next_path='J:\\video2.png'
#
# result = warp(prvs_path=prvs_path,next_path=next_path)
# cv2.imwrite('output-img/wrap.png',result)





import cv2
import numpy as np





# cap = cv2.VideoCapture('../data_video/video00.mp4')
#
# fps = cap.get(cv2.CAP_PROP_FPS)
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# out = cv2.VideoWriter('optflow_inter1a.mp4', fourcc, fps, (w, h))
#
# optflow_params = [0.5, 3, 15, 3, 5, 1.2, 0]
#
# frame_exists, prev_frame = cap.read()
# prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# y_coords, x_coords = np.mgrid[0:h, 0:w]
# coords = np.float32(np.dstack([x_coords, y_coords]))
# hsv=np.zeros_like(prev_frame)
# cv2.imwrite('f1.png',prev_frame)
#
#
# hsv[...,1]=255
# i=0
# while(cap.isOpened()):
#     frame_exists, curr_frame = cap.read()
#     if frame_exists:
#         curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
#         # cv2.imwrite('f2.png',curr_frame)
#         flow = cv2.calcOpticalFlowFarneback(curr, prev, None, *optflow_params)
#
#         # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         # hsv[...,0] = ang*180/np.pi/2
#         # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#         # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#         # cv2.imwrite('flw.png',rgb)
#         # cv2.imshow('frame2',hsv)
#
#         if i==6:
#            cv2.imwrite('f6.png',curr_frame)
#            inter_frames = interpolate_frames(prev_frame, coords, flow, 1)
#            for frame in inter_frames:
#               # out.write(frame)
#               cv2.imwrite('flow-2.png',frame)
#            break
#         i+=1
#         prev_frame = curr_frame
#         prev = curr
#     else:
#         break
#
# cap.release()
# out.release()
#



def interpolate_frames(frame, coords, flow, n_frames):
    frames = [frame]
    # for f in range(1,n_frames):
    pixel_map = coords + (1/n_frames) * flow
    inter_frame = cv2.remap(frame, pixel_map, None, cv2.INTER_LINEAR)
    frames.append(inter_frame)
    return frames

cap = cv2.VideoCapture('../data_video/video00.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
optflow_params = [0.5, 3, 15, 3, 5, 1.2, 0]
y_coords, x_coords = np.mgrid[0:h, 0:w]
coords = np.float32(np.dstack([x_coords, y_coords]))

prev=cv2.imread('f1.png')
curr=cv2.imread('f6.png')
hsv=np.zeros_like(prev)
# cv2.imwrite('f1.png',prev_frame)
#
#
hsv[...,1]=255


prev_frame=  cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
curr_frame =  cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, *optflow_params)

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imwrite('flw1-6.png',rgb)

inter_frames = interpolate_frames(prev, coords, flow, 1)

for frame in inter_frames:
    cv2.imwrite('f1-6.png',frame)
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('cha.png',frame-prev_frame)




# import cv2
# import numpy as np
# import time
# # load image
# prev = cv2.imread('f1.png')
# next = cv2.imread('f2.png')
#
# # change RGB to gray
# prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
#
#
#
# # calculate optical flow
# flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
# # calculate mat
# w = int(prev.shape[1])
# h = int(prev.shape[0])
# y_coords, x_coords = np.mgrid[0:h, 0:w]
# coords = np.float32(np.dstack([x_coords, y_coords]))
#
# t1=time.time()
# pixel_map = coords + flow
# new_frame = cv2.remap(prev, pixel_map, None, cv2.INTER_LINEAR)
# new_frame=cv2.copyMakeBorder(new_frame,10,10,10,10,cv2.BORDER_REPLICATE)
# print('time: %d'%((time.time()-t1)*1000))
#
# cv2.imwrite('new_frame.png', new_frame)