'''This script is used for checking the distribution of the images.
This is used to cheeck for imbalances in the training set.

'''
import pandas as pd
from glob import glob
import numpy as np
from collections import Counter
from utils.meta import *
info_files =  glob("trainval/*/*bbox.bin")


info_dict = {}
class_counter = Counter()
for bbox_file in info_files:
    bbox = np.fromfile(bbox_file, dtype=np.float32)
    info_dict[bbox_file] = int(bbox[9])
    class_counter[int(bbox[9])] += 1

    
for i in range(len(class_counter)):
    print(CLASSES_MAP[i], class_counter[i])
