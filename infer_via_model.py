
#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[90]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import glob
import pickle
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# get_ipython().run_line_magic('matplotlib', 'inline')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/ubuntu/data/sdc_label_map.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
    return cv2.imread(image)

PATH_TO_TEST_IMAGES_DIR = 'test'
TEST_IMAGE_PATHS = sorted(glob.glob("test/**/*.jpg"))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_images(images, graph):
    print('Inference')
    with graph.as_default():
        with tf.Session() as sess:
            output_dict_array = []
            for i, img in enumerate(images):
                image = img[0]
                file_path = img[1]
                print(i)
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                output_dict['path'] = file_path
                output_dict_array.append(output_dict)
    return output_dict_array

classes = []
images = []

for i, test_img in enumerate(TEST_IMAGE_PATHS):
  if i % 100 == 0:
    print ("Loading %d" %i)
  images.append(load_image_into_numpy_array(test_img))


output_dicts = run_inference_for_images(zip(images, TEST_IMAGE_PATHS), detection_graph)

pickle.dump(output_dicts, open('output.pkl', 'wb'))
l = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,0,0,0,0,0,0,0,0]
labels_dict = dict(zip([i for i in range(1, 23)], l))

classes = []
for d in output_dicts:
    if d['detection_classes'][np.argmax(d['detection_scores'])] > 0.2:
        classes.append((d['path'], d['detection_classes'][0], labels_dict[d['detection_classes'][0]]))
    else:
        classes.append((d['path'], 0, 0))

# classes1 = []
# for d1 in output_dicts_1:
#     if d1['detection_classes'][np.argmax(d1['detection_scores'])] > 0.6:
#         classes1.append(d1['detection_classes'][np.argmax(d1['detection_scores'])])
#     else:
#         classes1.append(0)
        
# classes2 = []
# for d2 in output_dicts_2:
#     if d2['detection_classes'][np.argmax(d2['detection_scores'])] > 0.6:
#         classes2.append(d2['detection_classes'][np.argmax(d2['detection_scores'])])
#     else:
#         classes2.append(0)

df = pd.DataFrame(classes)
df.to_csv('output_total.csv')



# from PIL import Image, ImageStat
# import math

# def brightness( im):
# #   im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    r,g,b = stat.mean
#    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

# def dark_or_night():
# #    np_file = list(np.load(file_name))
#     org_test_files = glob.glob(r"C:\Users\Shubham\Desktop\deploy\trainval\**\*.jpg")
#     im1 = [] 
#     for i, ele in enumerate(org_test_files):
#         print(i)
#         ele1 = Image.open(ele)
#         im1.append(brightness(ele1))
#     im1 = np.array(im1).reshape(1, -1)
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(im1.T)
#     return kmeans

