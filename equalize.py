'''This script applies adaptive histogram equalization to alll the
trainval/testval images.

Hand modified cause pulling in arguments is a pain and this will be
used twice to modify the images.
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage import exposure
from glob import glob
from multiprocessing import Pool


def equalize_img(img_file):
    img = skimage.io.imread(img_file)
    img_eq = exposure.equalize_adapthist(img)
    skimage.io.imsave(img_file, img_eq)


if __name__ == "__main__":
    img_files = glob("trainval/*/*.jpg")
    with Pool(4) as p:
        p.map(equalize_img, img_files)
