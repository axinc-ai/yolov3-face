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

if len(sys.argv)!=3:
	print("python annotation.py [fddb/medical-mask-dataset/mixed] [dataset folder path]")
	sys.exit(1)

MODE=sys.argv[1]
DATASET_ROOT_PATH=sys.argv[2]

if MODE!="fddb" and MODE!="medical-mask-dataset" and MODE!="mixed":
	print("Unknown mode "+MODE)
	sys.exit(1)

if(not os.path.exists(DATASET_ROOT_PATH)):
	print("folder not found "+DATASET_ROOT_PATH)
	sys.exit(1)

annotation_path="./train_"+MODE+".txt"
f_annotation=open(annotation_path,mode="w")

def fddb(f_annotation,root_src_dir,category):
	if(not os.path.exists(root_src_dir)):
		print("folder not found "+root_src_dir)
		sys.exit(1)

	for list in range(1,11):
		list2=str(list)
		if list<10:
			list2="0"+str(list)
		path=root_src_dir+"FDDB-folds/FDDB-fold-"+str(list2)+"-ellipseList.txt"
		lines=open(path).readlines()

		line_no=0

		while True:
			if line_no>=len(lines):
				break

			line=lines[line_no]
			line_no=line_no+1

			file_path=line.replace("\n","")
			image_path=root_src_dir+"originalPics/"+file_path+".jpg"

			image=cv2.imread(image_path)
			imagew=image.shape[1]
			imageh=image.shape[0]

			f_annotation.write(image_path+" ")
			
			line_n=int(lines[line_no])
			line_no=line_no+1

			for i in range(line_n):
				line=lines[line_no]
				line_no=line_no+1
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

				xmin=int(x-w/2)
				ymin=int(y-h/2)
				xmax=int(x+w/2)
				ymax=int(y+h/2)

				x=1.0*x/imagew
				y=1.0*y/imageh
				w=1.0*w/imagew
				h=1.0*h/imageh

				if w>0 and h>0 and x-w/2>=0 and y-h/2>=0 and x+w/2<=1 and y+h/2<=1:
					f_annotation.write(""+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+","+str(category)+" ")
				else:
					print("Invalid position removed "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
			
			f_annotation.write("\n")

def medical_mask_dataset(f_annotation,root_src_dir):
	if(not os.path.exists(root_src_dir)):
		print("folder not found "+root_src_dir)
		sys.exit(1)

	for src_dir, dirs, files in os.walk(root_src_dir):
		for file_ in files:
			root, ext = os.path.splitext(file_)

			if file_==".DS_Store":
				continue
			if file_=="Thumbs.db":
				continue
			if not(ext == ".txt"):
				continue
			if file_=="train.txt":
				continue
			
			path = src_dir + file_
			lines=open(path).readlines()
			print(path)

			jpg_path = file_.replace(".txt",".jpg")
			f_annotation.write(root_src_dir+jpg_path+" ")

			image=cv2.imread(root_src_dir+jpg_path)
			imagew=image.shape[1]
			imageh=image.shape[0]

			for line in lines:
				if line=="\n":
					continue
				data = line.split(" ")
				xmin=float(data[1])-float(data[3])/2
				ymin=float(data[2])-float(data[4])/2
				xmax=xmin+float(data[3])
				ymax=ymin+float(data[4])
				xmin=int(xmin*imagew)
				ymin=int(ymin*imageh)
				xmax=int(xmax*imagew)
				ymax=int(ymax*imageh)
				category=int(data[0])
				f_annotation.write(""+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+","+str(category)+" ")
			f_annotation.write("\n")

if MODE=="fddb":
	fddb(f_annotation,DATASET_ROOT_PATH+"fddb/",0)
if MODE=="medical-mask-dataset":
	medical_mask_dataset(f_annotation,DATASET_ROOT_PATH+"medical-mask-dataset/")
if MODE=="mixed":
	fddb(f_annotation,DATASET_ROOT_PATH+"fddb/",2)
	medical_mask_dataset(f_annotation,DATASET_ROOT_PATH+"medical-mask-dataset/")

f_annotation.close()
