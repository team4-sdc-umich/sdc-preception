'''This annotates all the images with the bounding boxes.

FIXME: Some bounding boxes are messed up but is proper in the tf
record files.

'''
import cv2 as cv
import pandas as pd
from glob import glob
from utils.meta import annotate_images
import os

img_files = sorted(glob("trainval/*/*.jpg"))

for cnt, img_file in enumerate(img_files):
    img = annotate_images(img_file)
    save_path = img_file.replace("trainval", "new")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv.imwrite(save_path, img)
    if cnt % 100 == 0:
        print ("Iteration at", cnt)
    

