from glob import glob
import numpy as np
import csv

files = glob('../deploy/test/*/*_image.jpg')#Specify test images path
inputFile = open('inputTask2.csv', 'r', newline='')
centroids = [row for row in csv.reader(inputFile)]
print('No. of files: ',len(files),'No. of centroids: ',len(centroids))
if(len(centroids)!=len(files)):
	print('No. of centroids does not match with the No. of files')
	exit()


files = [ [x[-2], x[-1].replace('_image.jpg','')] for x in [file.split('/')[-1].split('\\')[-2:] for file in files]]

outputFile = open('outputTask2.csv', 'w', newline='')
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['guid/image/axis','value'])

for i in range(len(files)):
	name = '/'.join(files[i])
	[x,y,z] = centroids[i]
	csvWriter.writerow([name+'/x',x])
	csvWriter.writerow([name+'/y',y])
	csvWriter.writerow([name+'/z',z])
	
inputFile.close()
outputFile.close()
	

