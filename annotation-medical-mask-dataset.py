import sys
import time
import argparse
import os

import cv2

root_src_dir="./medical-mask-dataset/"

annotation_path=root_src_dir+"/train.txt"
f_annotation=open(annotation_path,mode="w")

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
            #print(line)
            data = line.split(" ")
            #print(data)
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
