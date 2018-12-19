'''This script is used for checking if images are dark or not.  It
uses a day night classifier that was based on brightness statistics.
'''

import sklearn
from sklearn.externals import joblib
import pickle
from glob import glob
from PIL import Image, ImageStat
import numpy as np
import math

def brightness(im):
    #   im = Image.open(im_file)
    stat = ImageStat.Stat(im)
    r,g,b = stat.mean
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

dn_classifier_file = 'extras/day_night_classifier.pkl'
model = joblib.load(dn_classifier_file)

test_imgs = sorted(glob("train/*/*.jpg"))

brightness_images = []
for i, test_img in enumerate(test_imgs):
    print("Iteration", i)
    img = Image.open(test_img)
    stats = brightness(img)
    brightness_images.append(stats)


barr = np.array(brightness_images)
barr = barr.reshape(1, -1).T
dn_c = model.predict(barr)
