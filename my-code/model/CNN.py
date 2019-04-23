from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
from keras.utils import plot_model


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

def myloss(y_true, y_pred):
    # return 1- (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
    return K.sqrt(K.sum(K.square(y_true-y_pred)))

def CNN(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    img_input = Input(input_shape)

    x = MaxPooling2D()(img_input)
    x = Conv2D(64, ( 3,3), padding='same', name='con1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(128, ( 3,3), padding='same', name='conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2D(512, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Classifying layers.
    x = Conv2D(num_classes, ( 1, 1), strides=(1, 1), activation='linear')(x)
    x = BatchNormalization()(x)            # batchsize 256 512 19 361


    x = Lambda(lambda x: tf.image.resize_images(x, [512, 1024]))(x)

    x = Activation('softmax')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss= 'categorical_crossentropy',
                  metrics=[dice_coef])
    # plot_model(model, to_file='../model_train_result/CNN.png')
    return model
