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
import equiangular as eq
import random

ANGLE_RANGE=30

def plot_results(annotations, img, category):
	for annote in annotations:
		top_left = (annote[0], annote[1])
		bottom_right = (annote[2], annote[3])

		# update image
		color = (255, 40, 40)
		cv2.rectangle(img, top_left, bottom_right, color, 4)
	return img

if len(sys.argv)!=2:
	print("python annotation.py [folder path]")
	sys.exit(1)

DATASET_ROOT_PATH=sys.argv[1]

if(not os.path.exists(DATASET_ROOT_PATH)):
	print("folder not found "+DATASET_ROOT_PATH)
	sys.exit(1)

OUTPUT_LABEL="annotations_yolov3_keras"
OUTPUT_BASE_PATH=DATASET_ROOT_PATH+OUTPUT_LABEL

#TRAIN_PATH=OUTPUT_BASE_PATH+"/train.txt"
#TEST_PATH=OUTPUT_BASE_PATH+"/test.txt"

if(not os.path.exists(OUTPUT_BASE_PATH)):
	os.mkdir(OUTPUT_BASE_PATH)

#f_train=open(TRAIN_PATH,mode="w")
#f_test=open(TEST_PATH,mode="w")

file_no=0

annotation_path=OUTPUT_BASE_PATH+"/train.txt"
f_annotation=open(annotation_path,mode="w")

for list in range(1,11):
	list2=str(list)
	if list<10:
		list2="0"+str(list)
	path=DATASET_ROOT_PATH+"FDDB-folds/FDDB-fold-"+str(list2)+"-ellipseList.txt"
	lines=open(path).readlines()

	line_no=0

	while True:
		if line_no>=len(lines):
			break

		line=lines[line_no]
		line_no=line_no+1

		file_path=line.replace("\n","")
		image_path=DATASET_ROOT_PATH+"originalPics/"+file_path+".jpg"

		file_no=file_no+1

		image=cv2.imread(image_path)
		#print(image.shape)
		imagew=image.shape[1]
		imageh=image.shape[0]

		line_n = int(lines[line_no])
		for rot in range(-45, 46, 45):
			rot += random.randint(-10, 10)
			copy_path = OUTPUT_BASE_PATH + "/" + str(file_no) + "_" + str(rot) + ".jpg"
			invalid_annotation = False
			annotations = []
			_line_no = line_no + 1

			for i in range(line_n):
				line=lines[_line_no]
				_line_no=_line_no+1
				data=line.split(" ")
				major_axis_radius=float(data[0])
				minor_axis_radius=float(data[1])
				angle=float(data[2])
				center_x=float(data[3])
				center_y=float(data[4])

				x=center_x
				y=center_y
				w=minor_axis_radius*2
				h=major_axis_radius*2

				category=0
				eRect = eq.getEquiangularRect(rot, ANGLE_RANGE, int(x - w / 2), int(y - h / 2), w, h, imagew, imageh)
				xmin, ymin, xmax, ymax = eq.getMinimumEnclosingRect(eRect)
				xmin = int(xmin)
				ymin = int(ymin)
				xmax = int(xmax)
				ymax = int(ymax)

				if xmax - xmin > 0 and ymax - ymin > 0 \
					 and xmin >= 0 and ymin >= 0 \
					 and xmax < imagew and ymax < imageh:
					annotations.append((xmin, ymin, xmax, ymax, category))
				else:
					invalid_annotation = True
					print("Invalid position removed "+str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax))
					break
			
			if not invalid_annotation:
				f_annotation.write(copy_path + " ")
				for annote in annotations:
					f_annotation.write(""+str(annote[0])+","+str(annote[1])+","+str(annote[2])+","+str(annote[3])+","+str(annote[4])+" ")
				
				eImage = eq.getEquiangularImage(image, rot, ANGLE_RANGE)
				# eImage = plot_results(annotations, eImage, 0)
				# cv2.imshow(str(rot), eImage)
				# cv2.waitKey(0)
				cv2.imwrite(copy_path, eImage)
				f_annotation.write("\n")

		line_no = line_no + line_n + 1

f_annotation.close()
#f_train.close()
#f_test.close()
