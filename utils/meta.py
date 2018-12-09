import pandas as pd
import numpy as np
import os
import PIL
import io
import tensorflow as tf
import random

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


CLASSES_MAP = {
    0: 'Unknown',
    1: 'Compacts',
    2: 'Sedans',
    3: 'SUVs',
    4: 'Coupes',
    5: 'Muscle',
    6: 'SportsClassics',
    7: 'Sports',
    8: 'Super',
    9: 'Motorcycles',
    10: 'OffRoad',
    11: 'Industrial',
    12: 'Utility',
    13: 'Vans',
    14: 'Cycles',
    15: 'Boats',
    16: 'Helicopters',
    17: 'Planes',
    18: 'Service',
    19: 'Emergency',
    20: 'Military',
    21: 'Commercial',
    22: 'Trains'
}


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
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


def get_img_info(snapshot):
    ''' Extracts image info for the file path provided
    get_img_info :: Img -> InfoDict

    InfoDict has the following keys.
    path :: path of the file
    xmin :: bbox xmin 
    ymin :: bbox ymin 
    xmax :: bbox xmax 
    ymax :: bbox ymax 
    class :: class_num
    class_name :: class_name
    '''

    width, height = 0, 0

    with tf.gfile.GFile(snapshot, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        width, height = image.size

        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')

    assert width != 0
    assert height != 0


    if os.path.exists(snapshot.replace('_image.jpg', '_bbox.bin')):
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    else:
        print ("There's no bounding box for this: ", snapshot)
        return None


    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])


    bbox = bbox.reshape([-1, 11])
    
    vert_2D_list = []
    vert_2D = []

    for b in bbox:
        sz = b[6:9] 
        R = rot(b[0:3])
        t = b[3:6]
    
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]
        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]
        vert_2D_list.append(vert_2D)

    assert(len(vert_2D_list) <= 1)

    xmin = np.min(vert_2D[0, :])
    xmax = np.max(vert_2D[0, :])

    ymin = np.min(vert_2D[1, :])
    ymax = np.max(vert_2D[1, :])

    cls = int((bbox[0][-2]))
    assert(cls != 0)

    cls_name = CLASSES_MAP[cls]
    return {
        'width': width,
        'height': height,
        'path': snapshot,
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'class': cls,
        'name': cls_name,
        'format': b'jpeg',
        'encoded': encoded_jpg
    }

# def gen_table(trainval_path='trainval'):
#     return list(zip(sorted(glob(os.path.join(trainval_path, './*/*image.jpg'))),
#                     sorted(glob(os.path.join(trainval_path, './*/*proj.bin')))))
