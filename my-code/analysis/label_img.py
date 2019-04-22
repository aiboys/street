import cv2
import numpy as np
import matplotlib as plt
with open('../define/classes.txt','r') as f:
    CLASSES= f.read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("str") for c in CLASSES]
    COLORS = np.array(COLORS, dtype="str")
    f.close()
with open('../define/colors.txt','r') as f:
    # if a colors file was supplied, load it from disk
   if f:
        COLORS = f.read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")
        # print ('All COLORS: \n{0}'.format(COLORS))
       # otherwise, we need to randomly generate RGB colors for each class
       # label
   else:
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        # initialize the legend visualization
legend = 255*np.ones(((len(CLASSES) * 25) , 250, 3), dtype="uint8")
# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
        # draw the class name + color on the legend
        color = [int(c) for c in color]

        cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color[::-1]), -1)
        cv2.putText(legend, className, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # cv2.imshow('ok.png',legend)
        # k=cv2.waitKey(1)
        # if k==27:
        #      break

cv2.imwrite('legend.png',legend)
f.close()