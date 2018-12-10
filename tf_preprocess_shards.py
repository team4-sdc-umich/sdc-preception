''' This is to convert over train and test 
'''

import sys
sys.path.append('/home/akash/learn/tools/models/research')
sys.path.append('/home/akash/learn/tools/models/research/slim')

import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image
import utils.meta as meta
from sklearn.model_selection import train_test_split
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
import random
from glob import glob
import os
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util


flags = tf.app.flags
tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'data. Folder structure is assumed to be:'
                           '<data_dir>/training/label_2 (annotations) and'
                           '<data_dir>/data_object_image_2/training/image_2'
                           '(images).')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files'
                           'will be written. The TFRecord with the training set'
                           'will be located at: <output_path>_train.tfrecord.'
                           'And the TFRecord with the validation set will be'
                           'located at: <output_path>_val.tfrecord')
tf.app.flags.DEFINE_string('label_map_path', 'data/kitti_label_map.pbtxt',
                           'Path to label map proto.')
tf.app.flags.DEFINE_integer('validation_set_size', '500', 'Number of images to'
                            'be used as a validation set.')
tf.app.flags.DEFINE_integer('seed', '0', 'Seed for shuffling')
FLAGS = flags.FLAGS



def create_tf_example(info):

    # TODO(user): Populate the following variables from your example.

    height = info['height'] # Image height
    width =  info['width']# Image width
    filename = info['path'] # Filename of the image. Empty if image is not from file
    encoded_image_data = info['encoded'] # Encoded image bytes
    image_format = info['format']

    xmins = [info['xmin'] / width] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [info['xmax'] / width] # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [info['ymin'] / height] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [info['ymax'] / height] # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [info['name'].encode('utf8')] # List of string class name of bounding box (1 per box)
    classes = [info['class']] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    data_dir=FLAGS.data_dir,
    output_path=FLAGS.output_path,
    # classes_to_use=FLAGS.classes_to_use.split(','),
    label_map_path=FLAGS.label_map_path,
    validation_set_size=FLAGS.validation_set_size  
    seed = FLAGS.seed
    random.seed(seed)

    # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # train_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%
    #                                            output_path)
    # val_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord'%
    #                                          output_path)    

    print("Data dir is", data_dir[0])
    img_files = glob(os.path.join(data_dir[0], './*/*image.jpg'))
    random.shuffle(img_files)

    num_images = len(img_files)

    train_size = num_images - validation_set_size

    train_images = img_files[0: train_size]
    val_images = img_files[-validation_set_size:]



    num_shards=10

    base_dir = output_path[0]
    train_dir = os.path.join(output_path[0], 'train')
    val_dir = os.path.join(output_path[0], 'val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)        
        

        
    output_filebase=  os.path.join(output_path[0], 'train' ,'train_dataset.record')
    
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)
        for index, img_path in enumerate(train_images):
            info = meta.get_img_info(img_path)
            tfr = create_tf_example(info)            
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tfr.SerializeToString())


    output_filebase=  os.path.join(output_path[0], 'val','val_dataset.record')


    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)
        for index, img_path in enumerate(val_images):
            info = meta.get_img_info(img_path)
            tfr = create_tf_example(info)            
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tfr.SerializeToString())
            


    # train_writer.close()
    # val_writer.close()


if __name__ == '__main__':
  tf.app.run()
