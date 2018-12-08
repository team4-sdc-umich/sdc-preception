# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 19:37:58 2018

@author: Shubham
"""


import numpy as np
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

# def find_files(ext):
#     res = []
#     files = glob('deploy/*/*/*_' + ext)
#     if ext == 'bbox.bin':
#         for i, ele in enumerate(files):
#             bbox = np.fromfile(ele, dtype=np.float32)
#             bbox = bbox.reshape([-1, 11])
#             res.append(bbox)
#     elif ext == 'proj.bin':
#         for i, ele in enumerate(files):
#             proj = np.fromfile(ele, dtype=np.float32)
#             proj = proj.reshape([3, 4])
#             res.append(proj)
#     else:
#         for i, ele in enumerate(files):
#             cloud = np.fromfile(ele, dtype=np.float32)
#             cloud = cloud.reshape([3, -1])
#             res.append(cloud)
#     return res


def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e


# b_val = find_files('bbox.bin')
# c_val = find_files('cloud.bin')
# p_val = find_files('proj.bin')

def proj_2d(index):

    num_objs = 1

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 1), dtype=np.float32)

    # "Seg" area here is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)


    filename = 'train_' +str(index)
    bbox = np.fromfile(os.path.join('deploy','Annotations',(filename+'_bbox.bin')), dtype=np.float32)
    bbox = bbox.reshape([-1, 11])

    proj = np.fromfile(os.path.join('deploy','Annotations',(filename+'_proj.bin')), dtype=np.float32)
    proj = proj.reshape([3, 4])



    for k, b in enumerate(bbox):
        R = rot(b[0:3])
        t = b[3:6]

        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]

        # vert_2D_proj.append(vert_2D)

        x1 = np.min(vert_2D[0,:])
        x2 = np.max(vert_2D[0,:])
        y2 = np.max(vert_2D[1,:])
        y1 = np.min(vert_2D[1,:])

        boxes = [x1, y1, x2, y2]
        gt_classes = (bbox[0][-2])
        seg_areas = ((x2 - x1 + 1) * (y2 - y1 + 1))


        return     {  'boxes' : boxes,
            'gt_classes': gt_classes,
            'gt_overlaps' : overlaps,
            'flipped' : False,
            'seg_areas' : seg_areas}


    # return vert_2D_proj, boxes, gt_classes, seg_areas

print(proj_2d(4500))
