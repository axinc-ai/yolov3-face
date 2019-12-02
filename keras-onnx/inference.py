import os
import sys
import inspect
import colorsys
import onnx
import numpy as np
import tensorflow as tf
import keras
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from keras.models import load_model

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class YOLO(object):
    def __init__(self,classes_path):
        self.classes_path = classes_path

        #self.score = 0.3
        #self.iou = 0.45
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.session = None
        self.final_model = None

        print("classes "+str(len(self.class_names)))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        K.set_learning_phase(0)

    @staticmethod
    def _get_data_path(name):
        path = os.path.expanduser(name)
        #if not os.path.isabs(path):
        #    yolo3_dir = os.path.dirname(inspect.getabsfile(yolo3))
        #    path = os.path.join(yolo3_dir, os.path.pardir, path)
        return "./"+path

    def _get_class(self):
        classes_path = self._get_data_path(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def detect_with_onnx(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])

        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        feed_f = dict(zip(['input_1', 'image_shape'],
                          (image_data, np.array([image.size[1], image.size[0]], dtype='float32').reshape(1, 2))))
        all_boxes, all_scores, indices = self.session.run(None, input_feed=feed_f)

        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices:
            out_classes.append(idx_[1])
            out_scores.append(all_scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(all_boxes[idx_1])

        font = ImageFont.truetype(font=self._get_data_path('../font/FiraMono-Medium.otf'),
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image


def detect_img(yolo, img_url, output_url, model_file_name):
    image = Image.open(img_url)

    import onnxruntime
    yolo.session = onnxruntime.InferenceSession(model_file_name)

    r_image = yolo.detect_with_onnx(image)
    n_ext = img_url.rindex('.')
    score_file = output_url
    r_image.save(score_file, "JPEG")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("python inference.py input_model.onnx classes.txt input_image.jpg output_image.jpg")
        exit(-1)

    model_file_name = sys.argv[1]

    target_opset = 10

    detect_img(YOLO(sys.argv[2]), sys.argv[3], sys.argv[4], model_file_name)
