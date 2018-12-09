import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as vu
from object_detection.protos import string_int_label_map_pb2 as pb
from object_detection.data_decoders.tf_example_decoder import TfExampleDecoder as TfDecoder
from google.protobuf import text_format


def visualize(tfrecords_filename, label_map=None):
    if label_map is not None:
        label_map_proto = pb.StringIntLabelMap()
        with tf.gfile.GFile(label_map, 'r') as f:
            text_format.Merge(f.read(), label_map_proto)
            class_dict = {}
            for entry in label_map_proto.item:
                class_dict[entry.id] = {'name': entry.name}
    sess = tf.Session()
    decoder = TfDecoder(label_map_proto_file=label_map, use_display_name=False)
    sess.run(tf.tables_initializer())
    for record in tf.python_io.tf_record_iterator(tfrecords_filename):
        example = decoder.decode(record)
        host_example = sess.run(example)
        scores = np.ones(host_example['groundtruth_boxes'].shape[0])
        vu.visualize_boxes_and_labels_on_image_array(
            host_example['image'],
            host_example['groundtruth_boxes'],
            host_example['groundtruth_classes'],
            scores,
            class_dict,
            max_boxes_to_draw=None,
            use_normalized_coordinates=True)
        plt.imshow(host_example['image'])
        plt.show()
