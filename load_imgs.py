'''Script to load image files onto a list of numpy arrays that are
reshaped and converted into an 8 bit numpy RGB array.
'''



import numpy as np
from glob import glob
import os
from pathlib import Path
from PIL import Image
from multiprocessing import Pool

import cv2



def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height,
                                              im_width,
                                              3)).astype(np.uint8)



test_imgs = glob("test/*/*.jpg")
lst = [] 
for i, test_img in enumerate(test_imgs):
    print("Iteration ", i)
    img = cv2.imread(test_img)
    lst.append(img)

# for cnt, test_img in enumerate(test_imgs):
#     if cnt % 100 == 0:
#         print("Iteration", cnt)
#     output_path = test_img.replace("test", "npy")
#     output_dir_path = os.path.dirname(output_path)
#     Path(output_dir_path).mkdir(parents=True, exist_ok=True)
#     npy_img = load_image_into_numpy_array(Image.open(test_img))
#     np.save(output_path, npy_img)
