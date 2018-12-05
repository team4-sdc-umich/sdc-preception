# '''annotate_tool.py is a simple tool that reads a csv and allows one
# to label images via a keypress

# Valid labels::
# 0 -> 'None'
# 1 -> 'cars'
# 2 -> 'bikes'

# '''
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import sys
import cv2
import os

path_prefix = "deploy/test"  # Path to test data
global_exit_key = 'H'
labels = ["None", "Cars", "Bikes"]
labels_idx = ['0', '1', '2']
global_exit = False
ax1 = []

eventg = []
d = {}

def press(event):
    global global_exit
    global global_exit_key

    key_pressed = event.key

    if key_pressed == global_exit_key:
        global_exit = True

    if key_pressed in labels_idx:
        d[axes[0].get_title()] = int(event.key)
        c.set_text(labels[int(key_pressed)])
        # axes[1].text(0.5, 0.5, "This is asasd")
        fig.canvas.draw_idle()


if __name__ == "__main__":

    if (len(sys.argv) < 2):
        print ("Usage python annotate_tool.py <csv_file>")
        exit()
    file_name = sys.argv[1]
    # file_name = os.path.abspath("res/test_annotations_1.csv")
    df = pd.read_csv(file_name)



    for  i, row in df[df.valid == 0].iterrows():

        ann_image = row['guid/image']
        image = 'deploy/test/' + row['guid/image'] + '_image.jpg'
        image_arr = plt.imread(os.path.abspath(image))
        fig, axes = plt.subplots(2,1, figsize=(100,100))

        fig.canvas.mpl_connect('key_press_event', press)
        axes[0].imshow(image_arr)
        ax1 = axes[0].set_title(ann_image)

        class0 = ['Unkown','Boats', 'Helicopters', 'Planes', 'Service', 'Emergency' ,'Military', 'Commercial','Trains']
        class1 = ['Compacts','Sedans','SUVs','Coupes','Muscle','SportsClassics','Sports','Super']
        class2 = ['Motorcycles','OffRoad','Industrial','Utility','Vans','Helicopters']

        s =''
        for c in class0:
            s=s+ c + ' 0\n'
        for c in class1:
            s=s+ c + ' 1\n'
        for c in class2:
            s=s+ c + ' 2\n'

        axes[1].text(0.1, 0.8, s, horizontalalignment='left', verticalalignment='top', transform=axes[1].transAxes)

        c = axes[1].text(0.5, 0.5, "Enter class")

        mng = plt.get_current_fig_manager()

        plt.show()

        if global_exit:
            logging.warning("Exiting annotation loop")
            break

    for (img_loc, val) in d.items():
        df.loc[df['guid/image'] == img_loc, 'label'] = val
        df.loc[df['guid/image'] == img_loc, 'valid'] = 1

    df.to_csv(os.path.join(os.path.join(file_name)))
