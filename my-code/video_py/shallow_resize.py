import tensorflow as tf
import numpy as np
from keras.layers.core import Lambda


f2 = np.squeeze(np.load('sahllow_feature.npy') ) # 599 64 128 19
shallow_feature_final = []
with tf.Session() as sess:
 np.save("shallow_feature.npy",tf.image.resize_images((tf.convert_to_tensor(f2[:,:,:,:])),[256, 512]).eval())