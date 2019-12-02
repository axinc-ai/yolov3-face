#ailia detector api sample

import ailia

import numpy
import tempfile
import cv2
import os
import urllib.request
import sys
import time

# settings
if len(sys.argv) < 3:
    print("python inference.py input_model.onnx classes.txt input_image.jpg output_image.jpg")
    exit(-1)

model_path = sys.argv[1]+".prototxt"
weight_path = sys.argv[1]
classes_path = sys.argv[2]
img_path = sys.argv[3]

with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

print("loading ...");

# detector initialize
env_id = ailia.get_gpu_environment_id()
categories = len(class_names)
detector = ailia.Detector(model_path, weight_path, categories, format=ailia.NETWORK_IMAGE_FORMAT_RGB, channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST, range=ailia.NETWORK_IMAGE_RANGE_U_FP32, algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3, env_id=env_id)

# load input image and convert to BGRA
img = cv2.imread( img_path, cv2.IMREAD_UNCHANGED )
if img.shape[2] == 3 :
    img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
elif img.shape[2] == 1 : 
    img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

print( "img.shape=" + str(img.shape) )

work = img
w = img.shape[1]
h = img.shape[0]

print("inferencing ...");

# compute
threshold = 0.2
iou = 0.45

cnt = 3
for i in range(cnt):
	start=int(round(time.time() * 1000))
	detector.compute(img, threshold, iou)
	end=int(round(time.time() * 1000))
	print("## ailia processing time , "+str(i)+" , "+str(end-start)+" ms")

# get result
count = detector.get_object_count()

print("object_count=" + str(count))

detector.set_input_shape(416,416)


for idx  in range(count) :
    # print result
    print("+ idx=" + str(idx))
    obj = detector.get_object(idx)
    print("  category=" + str(obj.category) + "[ " + class_names[obj.category] + " ]" )
    print("  prob=" + str(obj.prob) )
    print("  x=" + str(obj.x) )
    print("  y=" + str(obj.y) )
    print("  w=" + str(obj.w) )
    print("  h=" + str(obj.h) )
    top_left = ( int(w*obj.x), int(h*obj.y) )
    bottom_right = ( int(w*(obj.x+obj.w)), int(h*(obj.y+obj.h)) )
    text_position = ( int(w*obj.x)+4, int(h*(obj.y+obj.h)-8) )

    # update image
    cv2.rectangle( work, top_left, bottom_right, (0, 0, 255, 255), 4)
    cv2.putText( work, class_names[obj.category], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)

# save image
cv2.imwrite( sys.argv[4], work)
