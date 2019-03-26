import cv2
import os
import numpy as np
# 这个程序是用来将camvid数据集的帧labeled图像转化为cityscapes的label格式，以便后续利用精确度来做适应性阈值
import matplotlib.pyplot as plt





file_camvid_path="K:\\cmbic\\labeled_cavmid"
file_trans_path="K:\\cmbic\\LabeledApproved_full_trans"

# os.walk helps to find all files in directory.
x_paths=[]


def find_path(original_path):
  for (path, dirname, files) in os.walk(original_path):
    pth=os.path.join(path,''.join(dirname))
    for filename in files:
        x_paths.append(os.path.join(pth, filename))

  return x_paths


path=[]



path=find_path(file_camvid_path)

for i in range(len(path)):
    im=cv2.imread(path[i])
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im=cv2.resize(im,(512, 256), interpolation=cv2.INTER_AREA)
    new_img  = np.zeros_like(im)
    x,y,z=im.shape
    for xx in range(x):
        for yy in range(y):
            pixel=im[xx,yy]
            if (pixel[0]==128) and (pixel[1]==64) and (pixel[2]==128):   #road
                new_img[xx,yy,0]=128
                new_img[xx,yy,1]=64
                new_img[xx,yy,2]=128

            if (pixel[0]==0) and (pixel[1]==0) and (pixel[2]==192):      #sidewalk
                new_img[xx,yy,0]=244
                new_img[xx,yy,1]=35
                new_img[xx,yy,2]=232

            if (pixel[0]==128) and (pixel[1]==0) and (pixel[2]==0):    # building
                new_img[xx,yy,0]=70
                new_img[xx,yy,1]=70
                new_img[xx, yy, 2] = 70

            if (pixel[0] == 64) and (pixel[1] == 192) and (pixel[2] == 0):    # wall
                    new_img[xx, yy, 0] = 102
                    new_img[xx, yy, 1] = 102
                    new_img[xx, yy, 2] =  156

            if (pixel[0]==64) and (pixel[1]==64) and (pixel[2]==128):          # fence
                new_img[xx,yy,0]=190
                new_img[xx,yy,1]=153
                new_img[xx, yy, 2] = 153

            if (pixel[0]==192) and (pixel[1]==192) and (pixel[2]==128):         #pole
                new_img[xx,yy,0]=153
                new_img[xx,yy,1]=153
                new_img[xx, yy, 2] =153

            if (pixel[0]==0) and (pixel[1]==64) and (pixel[2]==64):            # trafficlight
                new_img[xx,yy,0]=250
                new_img[xx,yy,1]=170
                new_img[xx, yy, 2] = 30

            if (pixel[0]==192) and (pixel[1]==128) and (pixel[2]==128):               # signsymbol
                new_img[xx,yy,0]=220
                new_img[xx,yy,1]=220
                new_img[xx, yy, 2] =0

            if (pixel[0]==128) and (pixel[1]==128) and (pixel[2]==0):                 # tree
                new_img[xx,yy,0]=107
                new_img[xx,yy,1]=142
                new_img[xx, yy, 2] = 35

            if (pixel[0]==64 ) and (pixel[1]==192) and (pixel[2]==128):              # ground
                new_img[xx,yy,0]=81
                new_img[xx,yy,1]=0
                new_img[xx, yy, 2] = 81

            if (pixel[0]==128) and (pixel[1]==128) and (pixel[2]==128):               #sky
                new_img[xx,yy,0]=70
                new_img[xx,yy,1]=130
                new_img[xx, yy, 2] = 180

            if (pixel[0]==0) and (pixel[1]==128) and (pixel[2]==192):              #bicylist
                new_img[xx,yy,0]=255
                new_img[xx,yy,1]=0
                new_img[xx, yy, 2] = 0

            if (pixel[0]==64) and (pixel[1]==64) and (pixel[2]==0):               #person
                new_img[xx,yy,0]=220
                new_img[xx,yy,1]=20
                new_img[xx, yy, 2] = 60

            if (pixel[0]==192) and (pixel[1]==0) and (pixel[2]==192):             # motorcyclescooter
                new_img[xx,yy,0]=255
                new_img[xx,yy,1]=0
                new_img[xx, yy, 2] =0

            if (pixel[0]==64) and (pixel[1]==0) and (pixel[2]==128):                # car
                new_img[xx,yy,0]=0
                new_img[xx,yy,1]=0
                new_img[xx, yy, 2] = 142

            if (pixel[0]==192) and (pixel[1]==128) and (pixel[2]==192):             # truck_bus
                new_img[xx,yy,0]=0
                new_img[xx,yy,1]=0
                new_img[xx, yy, 2] = 70




            if (pixel[0] == 64) and (pixel[1] == 128) and (pixel[2] == 64):       #animal
                 new_img[xx, yy, 0] = 220
                 new_img[xx, yy, 1] = 20
                 new_img[xx, yy, 2] = 60


            if (pixel[0] == 192) and (pixel[1] == 0) and (pixel[2] == 128):       #archway  拱道---墙
                 new_img[xx, yy, 0] = 102
                 new_img[xx, yy, 1] = 102
                 new_img[xx, yy, 2] = 156

            if (pixel[0] == 0) and (pixel[1] == 128) and (pixel[2] == 64):      #bridge
                new_img[xx, yy, 0] = 0
                new_img[xx, yy, 1] = 0
                new_img[xx, yy, 2] = 142

            if (pixel[0] == 64) and (pixel[1] == 0) and (pixel[2] == 192):       #  cartluggagepram
                 new_img[xx, yy, 0] = 0
                 new_img[xx, yy, 1] = 0
                 new_img[xx, yy, 2] = 142

            if (pixel[0] == 192) and (pixel[1] == 128) and (pixel[2] == 64):       # child
                 new_img[xx, yy, 0] = 220
                 new_img[xx, yy, 1] = 20
                 new_img[xx, yy, 2] = 60

            if (pixel[0] == 128) and (pixel[1] == 128) and (pixel[2] == 64):       #misc-text
                 new_img[xx, yy, 0] = 0
                 new_img[xx, yy, 1] = 0
                 new_img[xx, yy, 2] = 0

            if (pixel[0] == 192) and (pixel[1] == 0) and (pixel[2] == 64):       #lanmkgsnondriv
                 new_img[xx, yy, 0] = 128
                 new_img[xx, yy, 1] = 64
                 new_img[xx, yy, 2] = 128

            if (pixel[0] == 128) and (pixel[1] == 64) and (pixel[2] == 64):       #othermoving
                 new_img[xx, yy, 0] =  255
                 new_img[xx, yy, 1] = 0
                 new_img[xx, yy, 2] = 0

            if (pixel[0] == 128) and (pixel[1] == 128) and (pixel[2] == 192):       #roadwhoulder
                 new_img[xx, yy, 0] = 0
                 new_img[xx, yy, 1] = 0
                 new_img[xx, yy, 2] = 0

            if (pixel[0] == 128) and (pixel[1] == 0) and (pixel[2] == 192):       #lanemkgsdriv
                 new_img[xx, yy, 0] = 128
                 new_img[xx, yy, 1] = 64
                 new_img[xx, yy, 2] = 128

            if (pixel[0] == 64) and (pixel[1] == 128) and (pixel[2] == 192):       #suvpickuptrack
                 new_img[xx, yy, 0] = 0
                 new_img[xx, yy, 1] = 0
                 new_img[xx, yy, 2] = 70

            if (pixel[0] == 0) and (pixel[1] == 0) and (pixel[2] == 64):       #traffic cone
                 new_img[xx, yy, 0] = 0
                 new_img[xx, yy, 1] = 0
                 new_img[xx, yy, 2] = 0

            if (pixel[0] == 192) and (pixel[1] == 64) and (pixel[2] == 128):       #train
                 new_img[xx, yy, 0] = 0
                 new_img[xx, yy, 1] = 0
                 new_img[xx, yy, 2] = 0

            if (pixel[0] == 64) and (pixel[1] == 0) and (pixel[2] == 64):       #tunnel
                 new_img[xx, yy, 0] = 150
                 new_img[xx, yy, 1] = 120
                 new_img[xx, yy, 2] = 90

            if (pixel[0] == 192) and (pixel[1] == 192) and (pixel[2] == 0):       #vegetationmisc
                 new_img[xx, yy, 0] = 0
                 new_img[xx, yy, 1] = 0
                 new_img[xx, yy, 2] = 0

    new_img=new_img[...,::-1]

    cc=file_trans_path+'\\'+str(i)+'.png'
    cv2.imwrite(cc, new_img)