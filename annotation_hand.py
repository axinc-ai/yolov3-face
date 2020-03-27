#Generate annotation data for yolov3

from scipy import io as spio
from datetime import datetime

import re
import numpy as np
import os
import shutil
import sys
import glob
import cv2

if len(sys.argv)!=2:
	print("python annotation_hand.py [f:/hand_detection/dataset/vivahand/detectiondata/train/]")
	sys.exit(1)

DATASET_ROOT_PATH=sys.argv[1]

if(not os.path.exists(DATASET_ROOT_PATH)):
	print("folder not found "+DATASET_ROOT_PATH)
	sys.exit(1)

OUTPUT_LABEL="annotations_yolov3_keras"
OUTPUT_BASE_PATH=DATASET_ROOT_PATH+OUTPUT_LABEL

if(not os.path.exists(OUTPUT_BASE_PATH)):
	os.mkdir(OUTPUT_BASE_PATH)

annotation_path=OUTPUT_BASE_PATH+"/train.txt"
print(annotation_path)

f_annotation=open(annotation_path,mode="w")
file_list = glob.glob(DATASET_ROOT_PATH+"posGt/*")

for path in file_list:
	path=path.replace("\\","/")
	print(path)

	lines=open(path).readlines()

	line_no=1

	file_path=path.split("/")
	file_path=file_path[len(file_path)-1].split(".")[0]
	image_path=DATASET_ROOT_PATH+"pos/"+file_path+".png"
	print(image_path)

	image=cv2.imread(image_path)
	imagew=image.shape[1]
	imageh=image.shape[0]

	f_annotation.write(image_path+" ")

	while True:
		if line_no>=len(lines):
			break

		line=lines[line_no]
		line_no=line_no+1

		data=line.split(" ")
		category=0#data[0]
		xmin=int(data[1])
		ymin=int(data[2])
		w=int(data[3])
		h=int(data[4])

		xmax=xmin+w
		ymax=ymin+h

		f_annotation.write(""+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+","+str(category)+" ")
	f_annotation.write("\n")

f_annotation.close()
