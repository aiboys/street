from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Deconvolution2D
import tensorflow as tf
from keras.utils import plot_model
import numpy as np




def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] - 1
    # initialize a variable to store total IoU in
    mean_iou = K.variable(0)

    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)

    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def FCN(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    img_input = Input(input_shape)

    # Block 1

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1_')(img_input)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu',name='act')(x)


    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2_')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu',name='act2')(x)

    x = MaxPooling2D(name='pool1')(x)


    # Block 2

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1_')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu',name='act3')(x)


    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2_')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu',name='act4')(x)

    x = MaxPooling2D(name='pool2')(x)

    # Block 3

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1_')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu',name='act5')(x)


    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2_')(x)
    x = BatchNormalization(name='bn6')(x)
    x = Activation('relu',name='act6')(x)
    # x = Lambda(lambda x: tf.image.resize_images(x, [256, 512]))(x)


    x=Conv2D(19, (1, 1), strides=(1, 1), activation='linear', name='shallow_point')(x)
    x = BatchNormalization(name='bn7')(x)
    x = Activation('relu',name='act7')(x)
    x = Lambda(lambda x: tf.image.resize_images(x, [512, 1024]), name='lam1')(x)


#----------------------------------------------------------------------------------------------


    x = Lambda(lambda x: tf.image.resize_images(x, [128, 256]), name='labm2')(x)
    # x=MaxPooling2D(name='pool1')(x)

    x = Conv2D(256, (3, 3),padding='same', name='block3_conv3_')(x)
    x = BatchNormalization(name='bn8')(x)
    x = Activation('relu',name='act8')(x)
    # x = Lambda(lambda x: tf.image.resize_images(x, [64, 128]))(x)

    block_3_out = MaxPooling2D(name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1_')(block_3_out)
    x = BatchNormalization(name='bn9')(x)
    x = Activation('relu',name='act9')(x)


    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2_')(x)
    x = BatchNormalization(name='bn10')(x)
    x = Activation('relu',name='act10')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3_')(x)
    x = BatchNormalization(name='bn11')(x)
    x = Activation('relu',name='act11')(x)

    block_4_out = MaxPooling2D(name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1_')(block_4_out)
    x = BatchNormalization(name='bn12')(x)
    x = Activation('relu',name='act12')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2_')(x)
    x = BatchNormalization(name='bn13')(x)
    x = Activation('relu',name='act13')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3_')(x)
    x = BatchNormalization(name='bn14')(x)
    x = Activation('relu',name='act14')(x)

    x = MaxPooling2D(name='pool5')(x)

    # Load pretrained weights.
    # if vgg_weight_path is not None:
    #     vgg16 = Model(img_input, x)
    #     vgg16.load_weights(vgg_weight_path, by_name=True)

    # Convolutinalized fully connected layer.
    x = Conv2D(4096, (7, 7), activation='relu', padding='same',name='123')(x)
    x = BatchNormalization(name='bn15')(x)
    x = Activation('relu',name='act15')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same',name='1234')(x)
    x = BatchNormalization(name='bn16')(x)
    x = Activation('relu',name='act16')(x)

    # Classifying layers.
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',name='12345')(x)
    x = BatchNormalization(name='bn17')(x)

    block_3_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',name='123456')(block_3_out)
    block_3_out = BatchNormalization(name='bn18')(block_3_out)

    block_4_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear',name='1234567')(block_4_out)
    block_4_out = BatchNormalization(name='bn19')(block_4_out)


    x = Lambda(lambda x: tf.image.resize_images(x, [32,64]))(x)
    x = Add()([x, block_4_out])
    x = Activation('relu',name='act17')(x)


    x = Lambda(lambda x: tf.image.resize_images(x, [64,128]))(x)
    x = Add()([x, block_3_out])
    x = Activation('relu',name='act18')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, [512,1024]))(x)

    x = Activation('softmax',name='act19')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[mean_iou])
    # plot_model(model, to_file='model_train_result/FCN0.png')
    return model
