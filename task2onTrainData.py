#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


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
        [ 0, 0, 1, 2, 7, 7],
        [ 1, 2, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

def inside_polygon(x, y, points):
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def get_pcloud_within_bbox(uv,bbox):
    uv = uv[:2,:]
    inBB = [True]*uv.shape[1]
    for i in range(len(inBB)):
        inBB[i] = inBB[i] and inside_polygon(uv[0,i],uv[1,i],bbox)

    tags = np.where(inBB)
    uv = uv[:,tags]

    return uv,tags

def get_centroid(pcloud,tags):
    x = np.mean(pcloud[0,tags])
    y = np.mean(pcloud[1,tags])
    z = np.mean(pcloud[2,tags])

    #x = np.median(pcloud[0,tags]) #doesn't work great
    #y = np.median(pcloud[1,tags])
    #z = np.median(pcloud[2,tags])
    return x,y,z

def getData():
    files = glob('../deploy/trainval/*/*_image.jpg')
    #files = glob('../deploy/trainval/bc097e0a-99bf-438d-9826-5ba0f228ea34/0005_image.jpg')
    idx = np.random.randint(0, len(files))
    snapshot = files[idx]
    print(snapshot)

    img = plt.imread(snapshot)

    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz = xyz.reshape([3, -1])

    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    try:
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)

    bbox = bbox.reshape([-1, 11])

    return img, xyz, proj, bbox

img, pcloud, proj, bbox = getData()
uv = proj @ np.vstack([pcloud, np.ones_like(pcloud[0, :])])
uv = uv / uv[2, :]

clr = np.linalg.norm(pcloud, axis=0)
fig1 = plt.figure(1, figsize=(16, 9))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.imshow(img)

colors = ['C{:d}'.format(i) for i in range(10)]


for k, b in enumerate(bbox):
    R = rot(b[0:3])
    t = b[3:6]

    bbSize = b[6:9]
    vert_3D, edges = get_bbox(-bbSize / 2, bbSize / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]

    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
    vert_2D = vert_2D / vert_2D[2, :]
    points = vert_2D[:2,:].T
    hull = ConvexHull(points)
    clr = colors[np.mod(k, len(colors))]
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color=clr)
    #for e in edges.T:
    #    ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)

bbox2D = [(points[vertex, 0], points[vertex, 1]) for vertex in hull.vertices]
uv,tags = get_pcloud_within_bbox(uv, bbox2D)
x,y,z = get_centroid(pcloud,tags)
print('estimated centroid: ',x,y,z, '; true centroid: ', t)


ax1.scatter(uv[0, :], uv[1, :], c=clr, marker='.', s=1)

ax1.axis('scaled')
fig1.tight_layout()


plt.show()
