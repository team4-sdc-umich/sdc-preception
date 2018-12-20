from glob import glob
import pickle
from utils.meta import *
import pandas as pd
import numpy as np
inferences = []
with open('experiments/final_four.pkl', "rb") as f:
    inferences = pickle.load(f)


def size_of_bounding_boxes(bbox):
    ymin, xmin, ymax, xmax = bbox
    return abs((ymax - ymin) * (xmax - xmin))

def length_of_bounding_box(bbox):
    return bbox[3]*IMG_WIDTH - bbox[1]*IMG_WIDTH    
    
csv_dict = {}
for inference in inferences:
    c = inference['detection_classes'][0:2]
    s = inference['detection_scores'][0:2]
    b = inference['detection_boxes'][0:2]

    bbox_sizes = list(map(size_of_bounding_boxes, b))
    bargmax = np.argmax(bbox_sizes)

    p = inference['path']

    if s[0] > 0.20:
        if (abs(s[0] - s[1]) < 0.05) and (bbox_sizes[1] - bbox_sizes[0]) > 0:
            print("Using %s instead of %s" %(c[1], c[0]))
            csv_dict[p] = compress_classification(c[1])
        else:
            csv_dict[p] = compress_classification(c[0])
    else:
        csv_dict[p] = 0

csv_pd = pd.Series(csv_dict).to_frame()
csv_pd.to_csv("experiments/latest.csv")
