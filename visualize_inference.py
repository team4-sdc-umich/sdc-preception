'''This script was used for visualizing the bounding boxes and their
corresponding class scores.

FIXME: Some of the values are hardcoded but is trivial to fix.
'''

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from utils.meta import *
from glob import glob

from skimage.measure import compare_ssim as ssim
import numpy as np
import cv2

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


# FIXME: This should be changed to the appropriate pickle file.
dicts = ['info/out.pkl', 'info/out1.pkl', 'info/out2.pkl']


e = []
for i, dict_file in enumerate(dicts):
    with open(dict_file, "rb") as input_file:
        f = pickle.load(input_file)
        print("Length of %s pickle file is %d" % (dict_file, len(f)))
        e +=(f)

files = sorted(glob(("test/*/*.jpg")))


for i, f in enumerate(files):
    if i < 1500:
        continue
    fig, axes = plt.subplots(2,1, figsize=(100,100))
    print("Iteration == ", i)
    img = plt.imread(f)
    axes[0].imshow(img)
    axes[0].set_title(f)
    x = list(zip(e[i]['detection_scores'][0:3],
                 list(map(lambda x: CLASSES_MAP[x],
                          e[i]['detection_classes'][0:3]))))
    x = str(x)
    axes[1].text(0.5, 0.5, str(x))
    plt.show()

csv_file = []
for i, f in enumerate(files):
    if dn_c[i] == 1:
        csv_file.append((f, 2))
    else:
        csv_file.append((f, compress_classification(e[i]['detection_classes'][0])))

# prev_img = None
# similarity_scores = []
# for i, f in enumerate(files):
#     print("Iteration == ", i)
#     curr_img = plt.imread(f)
#     if prev_img is not None:
#         sim_score = ssim(curr_img, prev_img, multichannel=True)
#         print("ssim", sim_score)
#         similarity_scores.append(sim_score)
#     prev_img = curr_img

# similarity_scores =  pickle.load(open("ssim_scores.pickle", "rb"))
# l = 4
# mask = np.ones(l) / l
# smooth = np.convolve(similarity_scores, mask)
