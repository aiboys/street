import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


# 设标签宽W，长H
def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景）
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,
                                                                              n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)

def per_class_iu(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)


def label_mapping(input, mapping):  # 主要是因为CityScapes标签里面原类别太多，这样做把其他类别转换成算法需要的类别（共19类）和背景（标注为255）
    output = np.copy(input)  # 先复制一下输入图像
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]  # 进行类别映射，最终得到的标签里面之后0-18这19个数加255（背景）
    return np.array(output, dtype=np.int64)  # 返回映射的标签

def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):  # 计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:  # 读取info.json，里面记录了类别数目，类别名称，标签映射方式等等。
        info = json.load(fp)
    num_classes = np.int(info['classes'])  # 读取类别数目，这里是19类，详见博客中附加的info.json文件
    print('Num classes', num_classes)  # 打印一下类别数目
    name_classes = np.array(info['label'], dtype=np.str)  # 读取类别名称，详见博客中附加的info.json文件
    mapping = np.array(info['label2train'], dtype=np.int)  # 读取标签映射方式，详见博客中附加的info.json文件
    hist = np.zeros((num_classes, num_classes))  # hist初始化为全零，在这里的hist的形状是[19, 19]

    image_path_list = join(devkit_dir, 'val.txt')  # 在这里打开记录验证集图片名称的txt
    label_path_list = join(devkit_dir, 'label.txt')  # 在这里打开记录验证集标签名称的txt
    gt_imgs = open(label_path_list, 'r').read().splitlines()  # 获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()  # 获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]  #获得验证集图像分割结果路径列表，方便直接读取

    for ind in range(len(gt_imgs)):  # 读取每一个（图片-标签）对
        pred = np.array(Image.open(pred_imgs[ind]))  # 读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]))  # 读取一张对应的标签，转化成numpy数组
        label = label_mapping(label, mapping)  # 进行标签映射（因为没有用到全部类别，因此舍弃某些类别），可忽略
        if len(label.flatten()) != len(pred.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  # 对一张图片计算19×19的hist矩阵，并累加
        if ind > 0 and ind % 10 == 0:  # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs


def main(args):
    compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)  # 执行计算mIoU的函数


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str,
                        help='directory which stores CityScapes val gt images')  # 设置gt_dir参数，存放验证集分割标签的文件夹
    parser.add_argument('pred_dir', type=str,
                        help='directory which stores CityScapes val pred images')  # 设置pred_dir参数，存放验证集分割结果的文件夹
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list',
                        help='base directory of cityscapes')  # 设置devikit_dir文件夹，里面有记录图片与标签名称及其他信息的txt文件
    args = parser.parse_args()
    main(args)  # 执行主函数




import xlwt as excel
from decimal import Decimal

# def cal_various_metrics(name,pre_img, label):
#     file = excel.Workbook(encoding='utf-8')
#     table1 = file.add_sheet(name)
#     table_raw = 0 # 表从第一行开始记录数据
#
#     # pre_img = cv2.imread(data_path + 'pre\\top_mosaic_09cm_area30_pre_pj.tif', cv2.IMREAD_GRAYSCALE)
#     # label = cv2.imread(data_path + 'label\\top_mosaic_09cm_area30_lb.tif', cv2.IMREAD_GRAYSCALE)
#     pre_img=pre_img
#     label=label
#
#
#     confu_mat = metrics.confusion_matrix(label.reshape(-1), pre_img.reshape(-1))
#     confu_mat = confu_mat.astype(np.float32) # int32-float32
#     raw, _ = confu_mat.shape
#     col_sum = np.sum(confu_mat, axis=1) # 按行求和
#     # print(col_sum)
#     for i in range(raw):
#        #print(col_sum[i])
#        confu_mat[i, :] = confu_mat[i, :]/col_sum[i]
#        # print(confu_mat)
#        # print(np.sum(confu_mat, axis=1))
#        # 将混淆矩阵写入excel中
#     # table1.write(table_raw, 0, 'confusion_matrix:0-1')
#     tp = []  # 每一类的精度
#     fn = np.sum(confu_mat, axis=0) # 实际预测结果中各类的错误判别（这里把各类的tp包括，后面减去）
#     for i in range(raw):
#        table_raw = table_raw + 1
#        tp.append(confu_mat[i, i])
#        for j in range(raw):
#            table1.write(table_raw, j, Decimal(float(confu_mat[i, j])).quantize(Decimal("0.000")))
#     # 计算f1-score
#
#     tp = np.array(tp)
#     fn = fn - tp
#     fp = 1 - tp
#     # 计算并写出precision，f1-score，f1-m以及mIOU
#     table_raw = table_raw + 1
#     table1.write(table_raw, 0, 'precision')
#     f1_m = []
#     iou_m = []
#     for i in range(raw):
#       # 写出precision
#       table1.write(table_raw+1, i, Decimal(float(tp[i])).quantize(Decimal("0.000")))
#       # 写出f1-score
#       f1 = tp[i]*2/(tp[i]*2 + fp[i] + fn[i])
#       # f1_m.append(f1)
#       iou = tp[i]/(tp[i] + fp[i] + fn[i])
#       iou_m.append(iou)
#       table1.write(table_raw+2, i, Decimal(float(f1)).quantize(Decimal("0.000")))
#     f1_m = np.array(f1_m)
#     table1.write(table_raw + 3, 0, 'f1-m')
#     table1.write(table_raw + 3, 1, Decimal(float(np.mean(f1_m))).quantize(Decimal("0.000")))
#     iou_m = np.array(iou_m)
#     table1.write(table_raw + 4, 0, 'mIOU')
#     table1.write(table_raw + 4, 1, Decimal(float(np.mean(iou_m))).quantize(Decimal("0.000")))
#     file.save(name + '.xls')
#     return np.mean(iou_m)