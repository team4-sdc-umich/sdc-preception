from glob import glob
import numpy as np
import csv

files = glob('../deploy/test/*/*_image.jpg') #Specify test images path
inputFile = open('inputTask1.csv', 'r', newline='') 
labels = [row for row in csv.reader(inputFile)]
print('No. of files: ',len(files),'No. of labels: ',len(labels))
if(len(labels)!=len(files)):
	print('No. of labels does not match with the No. of files')
	exit()


files = [ [x[-2], x[-1].replace('_image.jpg','')] for x in [file.split('/')[-1].split('\\')[-2:] for file in files]]

outputFile = open('outputTask1.csv', 'w', newline='')
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['guid/image','label'])

for i in range(len(files)):
	csvWriter.writerow(['/'.join(files[i]),*labels[i]])
	
inputFile.close()
outputFile.close()
	

