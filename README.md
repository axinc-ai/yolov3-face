# YoloV3 Face

Implement Face detection using keras-yolo3.

## Requirements

- Keras 2.2.4 or lower (Issue : https://github.com/qqwweee/keras-yolo3/issues/544)
- Tensorflow 1.13.2 or lower (Issue : https://github.com/NVIDIA/TensorRT/issues/339)
- keras2onnx 1.5.1 (1.5.2 failed to convert onnx)
- keras-yolo3 (checked on version Jul 31, 2018 e6598d13c703029b2686bc2eb8d5c09badf42992)
- Python 3.5 or later
- OpenCV

## Install

```
git submodule init
git submodule update
```

## Create dataset

### fddb

Download fddb dataset (FDDB-folds and originalPics folder) and put in the each folder.

http://vis-www.cs.umass.edu/fddb/

Folder layout examples.

```
/Volumes/ST5/dataset/fddb/FDDB-folds/*.txt
/Volumes/ST5/dataset/fddb/originalPics/2002/*
/Volumes/ST5/dataset/fddb/originalPics/2003/*
```

Create fddb annotation data.

```
python3 annotation.py fddb /Volumes/ST5/dataset/
```

Output is train_fddb.txt.

### medical-mask-dataset

Download medical mask dataset.

https://www.kaggle.com/vtech6/medical-masks-dataset?fbclid=IwAR0DJG_Ov8dGYWTFrI3VHp89S-LtYVDyKMnj5aCJZtPHasG2gonH3F1xuWo

Create medical-mask-dataset annotation data.

```
python3 annotation.py medical-mask-dataset /Volumes/ST5/dataset/
```

Output is train_medical-mask-dataset.txt.

### mixed

Create fddb + medical-mask-dataset annotation data.

```
python3 annotation.py mixed /Volumes/ST5/dataset/
```

Output is train_mixed.txt.

## Training

### fddb

Training from fddb 2845 pictures.

```
python3 train.py fddb ./model_data/face_classes.txt ./model_data/tiny_yolo_anchors.txt
```

This is an output data path.

```
./model_data/log/trained_weights_final.h5
```

### medical-mask-dataset

Trained from medical-mask-dataset 678 pictures.

```
python3 train.py medical-mask-dataset ./model_data/mask_classes.txt ./model_data/tiny_yolo_anchors.txt
```

### mixed

Trained from fddb + medical-mask-dataset 2845 + 678 pictures.

```
python3 train.py mixed ./model_data/mask_classes.txt ./model_data/tiny_yolo_anchors.txt
```

## Convert to ONNX

### fddb

```
cd keras-onnx
python3 keras-yolo3-to-onnx.py ../model_data/logs/trained_weights_final.h5 ../model_data/face_classes.txt ../model_data/tiny_yolo_anchors.txt ../model_data/ax_face.onnx
```

### medical-mask-dataset or mixed

```
cd keras-onnx
python3 keras-yolo3-to-onnx.py ../model_data/logs/trained_weights_final.h5 ../model_data/mask_classes.txt ../model_data/tiny_yolo_anchors.txt ../model_data/ax_masked_face.onnx
```

## Inference using ONNX Runtime

### fddb

```
cd keras-onnx
python3 inference.py ../model_data/ax_face.onnx ../model_data/face_classes.txt ../images/couple.jpg output.jpg
```

![Output](./keras-onnx/output.jpg)

### medical-mask-dataset or mixed

```
cd keras-onnx
python3 inference.py ../model_data/ax_masked_face.onnx ../model_data/mask_classes.txt ../images/couple.jpg output.jpg
```

## Convert to ailia SDK

Optimize onnx file and export prototxt file

### fddb

```
cd onnx-ailia
python3 onnx_optimizer.py --yolov3 ../model_data/ax_face.onnx
python3 onnx2prototxt.py ../model_data/ax_face.opt.onnx
```

### medical-mask-dataset or mixed

```
cd onnx-ailia
python3 onnx_optimizer.py --yolov3 ../model_data/ax_masked_face.onnx
python3 onnx2prototxt.py ../model_data/ax_masked_face.opt.onnx
```

## Inference using ailia SDK

Inference using detector API

### fddb

```
cd onnx-ailia
python3 inference.py ../model_data/ax_face.opt.onnx ../model_data/face_classes.txt ../images/couple.jpg output.jpg
```

### medical-mask-dataset or mixed

```
cd onnx-ailia
python3 inference.py ../model_data/ax_masked_face.opt.onnx ../model_data/mask_classes.txt ../images/couple.jpg output.jpg
```

## Reference

- keras-yolo3 : https://github.com/qqwweee/keras-yolo3
- keras2onnx : https://github.com/onnx/keras-onnx/tree/master/applications/yolov3
