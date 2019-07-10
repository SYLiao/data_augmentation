import sys
sys.path.append('/Users/shiyaoliao/Documents/jd_work/result')
import codecs
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import result.get_dataset_colormap as get_dataset_colormap
import cv2 as cv
import os
from result.product_in_hand_delete import check_in_hand, check_hand

LABEL_NAMES = np.asarray([
    'background', 'arm', 'hand', 'product_in_hand', 'product_on_table'
])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)
class DeepLabModel(object):
    INPUT_TENSOR_NAME1 = 'ImageTensor:0'
    # INPUT_TENSOR_NAME1 = 'image_input:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE_h = 400
    INPUT_SIZE_w = 400
    def __init__(self, model_path):
        self.graph = tf.Graph()
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)
    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (400, 400)
        resized_image = image.convert('RGB').resize(target_size,
                                                    Image.ANTIALIAS)
        tf.expand_dims(resized_image, 0)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={
                self.INPUT_TENSOR_NAME1: [np.asarray(image)],
                # self.INPUT_TENSOR_NAME2: [np.asarray(image)],
            })
        seg_map = batch_seg_map[0]
        return image, seg_map

def vis_segmentation(image, seg_map, file):
    seg_image = get_dataset_colormap.label_to_color_image(seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    cv.imwrite(object_path + '/' + file.split('.')[0] + '.jpg', seg_image)