import matplotlib.pyplot as plt
import csv

loss_img=[]
val_loss_img=[]
dice_coef_img=[]
val_dice_coef_img=[]

with open('histor_loss.csv','r') as f1:
    for i in csv.reader(f1):
        for ii in range(len(i)):
         loss_img.append(float(i[ii]))
f1.close()

with open('histor_val_loss.csv','r') as f2:
    for j in csv.reader(f2):
        for jj in range(len(j)):
          val_loss_img.append(float(j[jj]))
f2.close()

with open('histor_dice_coef.csv','r') as f3:
    for k in csv.reader(f3):
        for kk in range(len(k)):
         dice_coef_img.append(float(k[kk]))
f3.close()

with open('histor_val_dice_coef.csv','r') as f4:
    for l in csv.reader(f4):
        for ll in range(len(l)):
         val_dice_coef_img.append(float(l[ll]))
f4.close()



plt.title("loss")
plt.plot(loss_img[11:], color="r", label="train")
plt.plot(val_loss_img[11:], color="b", label="val")
plt.legend(loc="best")
plt.savefig('loss.png')

plt.gcf().clear()
plt.title("dice_coef")
plt.plot(dice_coef_img[11:], color="r", label="train")
plt.plot(val_dice_coef_img[11:], color="b", label="val")
plt.legend(loc="best")
plt.savefig('dice_coef.png')