# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:30:33 2018

@author: Shubham
"""

import shutil
import os
from os.path import join as pjoin
import glob

base_start_dir = r'C:\Users\Shubham\Desktop\vision\deploy\trainval'
base_target_dir = r'C:\Users\Shubham\Desktop\faster-rcnn.pytorch\data\EECS598_dataset\data\Annotations'

base_start_files = os.listdir(base_start_dir)

for ele in base_start_files:
    ext = '.jpg'
    ext1 = '_bbox.bin'
    img_folder = pjoin(base_start_dir, ele)
    if '.csv' in img_folder:
        print (".csv file found")
        continue
    else:
        try:
            l = os.listdir(img_folder)
            for file in l:
                if ext1 in file:
                    new_file_name = ele + '_' + file
                    os.rename(pjoin(pjoin(base_start_dir, ele), file), 
                              pjoin(pjoin(base_start_dir, ele), new_file_name))
                    source_file = pjoin(pjoin(base_start_dir, ele), new_file_name)
                    shutil.copy(source_file, base_target_dir)
        except:
            print("Not a bbox file")
            
            
