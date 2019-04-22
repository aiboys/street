
import tensorflow as tf
from keras import backend as K
from model.fcn import fcn_8s
import os
import argparse
from model.ASPP import ASPP
from model.aspp2 import ASPP2
from model.FCN_0 import FCN
from model.CNN import CNN
from model.with_cnn_deep import with_cnn_deep
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))


parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'ASPP', 'pspnet','CNN','ASPP2','FCN0','with_cnn_deep'],
                    help="Model to test. 'fcn', 'unet', 'pspnet' is available.")
args = parser.parse_args()
model_name = args.model

# if model_name == "fcn":
#  model = fcn_8s(input_shape=(256, 512, 3), num_classes=19, lr_init=3e-4, lr_decay=5e-4)
# elif model_name == "ASPP":
#     model = ASPP(input_shape=(256, 512, 3), num_classes=19, lr_init=3e-4, lr_decay=5e-4)
# elif model_name == "ASPP2":
#     model = ASPP2(input_shape=(256, 512, 3), num_classes=19, lr_init=3e-4, lr_decay=5e-4)
# elif model_name == "FCN0":
#     model = FCN(input_shape=(256, 512, 3), num_classes=19, lr_init=3e-4, lr_decay=5e-4)
# elif model_name == "CNN":
#     model = CNN(input_shape=(256,512,38),num_classes=19,lr_init=3e-4,lr_decay=5e-4)
if model_name == "FCN0":
    model_deep = with_cnn_deep(input_shape=(256, 512, 19), num_classes=19)
try:
    model_deep.load_weights('./weight/' +model_name+ '_model_weight.h5',by_name=True)
except:
    print("You must train model and get weight before test.")


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


output_graph_name = './weight/' + 'fcn_deep_model_weight.pb'
output_fld = ''
# K.set_learning_phase(0)

print('input is :', model_deep.input.name)
print('output is:', model_deep.output.name)

sess = K.get_session()
frozen_graph = freeze_session(K.get_session(), output_names=[model_deep.output.op.name])

from tensorflow.python.framework import graph_io

tf.train.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', os.path.join(output_fld, output_graph_name))



