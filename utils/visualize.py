import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as vu
from object_detection.protos import string_int_label_map_pb2 as pb
from object_detection.data_decoders.tf_example_decoder import TfExampleDecoder as TfDecoder
from google.protobuf import text_format



global_exit_key = "H"
global_exit = False
d = {}
axes = []
c = []
fig = []
def press(event):
    global global_exit
    global global_exit_key
    global axes
    global c
    global fig
    key_pressed = event.key

    if key_pressed == global_exit_key:
        global_exit = True

    if event.key in ["0", "1", "2"]:
        d[axes[0].get_title()] = int(event.key)
        c.set_text(key_pressed)
        # axes[1].text(0.5, 0.5, "This is asasd")
        fig.canvas.draw_idle()


def visualize(tfrecords_filename, label_map=None):
    global axes
    global c
    global fig
    keep = 1
    class_dict = {}
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
        img_loc = host_example['source_id'].decode("utf-8")
        if img_loc in d:
            continue
        scores = np.ones(host_example['groundtruth_boxes'].shape[0])
        vu.visualize_boxes_and_labels_on_image_array(
            host_example['image'],
            host_example['groundtruth_boxes'],
            host_example['groundtruth_classes'],
            scores,
            class_dict,
            max_boxes_to_draw=None,
            use_normalized_coordinates=True)

        keep = 1


    
        d[img_loc] = keep
        fig, axes = plt.subplots(2, 1, figsize=(100, 100))
        fig.canvas.mpl_connect('key_press_event', press)
        axes[0].imshow(host_example['image'])
        ax1 = axes[0].set_title(img_loc)
        c = axes[1].text(0.5, 0.5, str(keep))
        mng = plt.get_current_fig_manager()
        plt.show()
        if global_exit:
            break


        return host_example
    

# scores = np.ones(host_example['groundtruth_boxes'].shape[0])
# vu.visualize_boxes_and_labels_on_image_array(
#             host_example['image'],
#             host_example['groundtruth_boxes'],
#             host_example['groundtruth_classes'],
#             scores,
#             class_dict,
#             max_boxes_to_draw=None,
#             use_normalized_coordinates=True)        
