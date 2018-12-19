import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv


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

bbox_data = pickle.load(open('C:/Users/spate/Downloads/output.pkl',"rb"))
#bbox_data1 = pickle.load(open('C:/Users/spate/Downloads/out1.pkl',"rb"))
#bbox_data2 = pickle.load(open('C:/Users/spate/Downloads/out2.pkl',"rb"))

detection_boxes = []
confidence = []
for i in range(len(bbox_data)):
    #print(bbox_data[i])
    detection_boxes.append(bbox_data[i]['detection_boxes'][np.argmax(bbox_data[i]['detection_scores'])])
    confidence.append(bbox_data[i]['detection_scores'][np.argmax(bbox_data[i]['detection_scores'])])

'''
for i in range(len(bbox_data1)):
    #print(bbox_data[i])
    detection_boxes.append(bbox_data1[i]['detection_boxes'][np.argmax(bbox_data1[i]['detection_scores'])])
    confidence.append(bbox_data1[i]['detection_scores'][np.argmax(bbox_data1[i]['detection_scores'])])

for i in range(len(bbox_data2)):
    #print(bbox_data[i])
    detection_boxes.append(bbox_data2[i]['detection_boxes'][np.argmax(bbox_data2[i]['detection_scores'])])
    #confidence.append(max(bbox_data2[i]['detection_scores']))
    confidence.append(bbox_data2[i]['detection_scores'][np.argmax(bbox_data2[i]['detection_scores'])])
    
'''

print(len(detection_boxes))

centroids = []

outputFile = open('outputTask2.csv', 'w', newline='')
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['guid/image/axis', 'value'])

#outputFile2 = open('outputTask2Sorted.csv', 'w', newLine='')
#csvWriter2 = csv.writer(outputFile2)
#csvWriter2.writerow(['guid/image/axis', 'value'])

#snapshot = 'C:/Users/spate/Downloads/EECS598(Proj)/deploy/test/0ff0a23e-5f50-4461-8ccf-2b71bead2e8e/0000_image.jpg'
images = glob.glob('C:/Users/spate/Downloads/EECS598(Proj)/deploy/test/**/*.jpg')
#img = plt.imread('C:/Users/spate/Downloads/EECS598(Proj)/deploy/test/0ff0a23e-5f50-4461-8ccf-2b71bead2e8e/0000_image.jpg')

for i in range(len(images)):

    img = plt.imread(images[i])
    names = images[i].split('\\')
    image_name = names[2].split('_')[0]
    print(names)
    print(image_name)

    pcloud = np.fromfile(images[i].replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    pcloud = pcloud.reshape([3, -1])

    proj = np.fromfile(images[i].replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    uv = proj @ np.vstack([pcloud, np.ones_like(pcloud[0, :])])
    uv = uv / uv[2, :]

    clr = np.linalg.norm(pcloud, axis=0)
    fig1 = plt.figure(1, figsize=(16, 9))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.imshow(img)
    #ax1.scatter(uv[0, :], uv[1, :], c=clr, marker='.', s=1)
    #ax1.axis('scaled')
    #fig1.tight_layout()

    #colors = ['C{:d}'.format(i) for i in range(10)]

    w,h = img.shape[0],img.shape[1]
    y1 = detection_boxes[i][0]*w
    x1 = detection_boxes[i][1]*h
    y2 = detection_boxes[i][2]*w
    x2 = detection_boxes[i][3]*h

    b_width = x2-x1   # Shall need for using the box height and width
    b_height = y2-y1

    #print(b_width)
    #print(b_height)

    bbox2D = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    #print(bbox2D)

    uv,tags = get_pcloud_within_bbox(uv, bbox2D)
    #print(tags)

    x,y,z = get_centroid(pcloud,tags)
    print("Estimated Centroid:",x,y,z)
    print(confidence[i])

    #if x is np.nan or y is np.nan or z is np.nan:
    #    x = -1.851861448
    #    y = -2.976851834
    #    z = 36.19718569
    #centroids.append((x,y,z))

    csvWriter.writerow([names[1]+'/'+image_name+'/x',x])
    csvWriter.writerow([names[1]+'/'+image_name+'/y',y])
    csvWriter.writerow([names[1]+'/'+image_name+'/z',z])

    ax1.plot([x1, x2], [y1, y1], color='r', linestyle='-', linewidth=2)
    ax1.plot([x2, x2], [y1, y2], color='r', linestyle='-', linewidth=2)
    ax1.plot([x1, x2], [y2, y2], color='r', linestyle='-', linewidth=2)
    ax1.plot([x1, x1], [y1, y2], color='r', linestyle='-', linewidth=2)

    ax1.scatter(uv[0, :], uv[1, :],c = 'blue', marker='.', s=1)

    ax1.axis('scaled')
    fig1.tight_layout()

    #plt.show()

    #num = input("Enter 1 for writing, 2 for abort")

    #if num == 1:
    #    csvWriter2.writerow([names[1] + '/' + image_name + '/x', x])
    #    csvWriter2.writerow([names[1] + '/' + image_name + '/y', y])
    #    csvWriter2.writerow([names[1] + '/' + image_name + '/z', z])

outputFile.close()
#outputFile2.close()



