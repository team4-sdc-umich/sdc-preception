import pickle
import numpy as np
import glob
import csv
import time
import matplotlib.pyplot as plt


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


bbox_data = pickle.load(open('output_30k.pkl',"rb"))

detection_boxes = []
confidence = []
for i in range(len(bbox_data)):
    detection_boxes.append(bbox_data[i]['detection_boxes'][np.argmax(bbox_data[i]['detection_scores'])])
    confidence.append(bbox_data[i]['detection_scores'][np.argmax(bbox_data[i]['detection_scores'])])

centroids = []

outputFile = open('outputTask2.csv', 'w', newline='')
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['guid/image/axis', 'value'])


images = glob.glob('../deploy/test/**/*.jpg')

for i in range(len(images)):
    t0 = time.time()
    img = plt.imread(images[i])
    names = images[i].split('\\')
    image_name = names[2].split('_')[0]
    
    pcloud = np.fromfile(images[i].replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    pcloud = pcloud.reshape([3, -1])

    proj = np.fromfile(images[i].replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    uv = proj @ np.vstack([pcloud, np.ones_like(pcloud[0, :])])
    uv = uv / uv[2, :]

    clr = np.linalg.norm(pcloud, axis=0)

    w,h = img.shape[0],img.shape[1]
    y1 = detection_boxes[i][0]*w
    x1 = detection_boxes[i][1]*h
    y2 = detection_boxes[i][2]*w
    x2 = detection_boxes[i][3]*h
    
    bbox2D = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    x_center = (x1+x2)/2.0
    y_center = (y1+y2)/2.0
    uv,tags = get_pcloud_within_bbox(uv, bbox2D)
    
    z = np.median(pcloud[2,tags])

    inv_proj = np.linalg.pinv(proj)
    est_centroids = (inv_proj @ np.array([[x_center],[y_center],[1]]))*z
    
    csvWriter.writerow([names[1]+'/'+image_name+'/x',*est_centroids[0]])
    csvWriter.writerow([names[1]+'/'+image_name+'/y',*est_centroids[1]])
    csvWriter.writerow([names[1]+'/'+image_name+'/z',*est_centroids[2]])

    print(i,'/', len(images), str((time.time()-t0)%60)[:8], 'seconds/iteration',end='\r')
    


outputFile.close()



