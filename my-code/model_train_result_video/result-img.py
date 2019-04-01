import matplotlib.pyplot as plt
import csv

loss_img=[]
val_loss_img=[]
dice_coef_img=[]
val_dice_coef_img=[]


class plot():
    def __init__(self,name,path):
        self.path=path
        self.name=name
    def csvread(self):
        file=[]
        with open(self.path,'r') as f:
            for i in csv.reader(f):
                for ii in range(len(i)):
                    file.append(float(i[ii]))
        f.close()
        print('%s finished'%(self.name))
        return file

file1=plot('loss','histor_loss.csv')
loss_img=file1.csvread()

file2=plot('val_loss','histor_val_loss.csv')
val_loss_img=file2.csvread()

file3=plot('dice_coef','histor_dice_coef.csv')
dice_coef_img=file3.csvread()

file4=plot('val_dice_coef','histor_val_dice_coef.csv')
val_dice_coef_img=file4.csvread()



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