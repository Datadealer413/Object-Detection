# %%

import numpy as np
import os

import tensorflow.compat.v2 as tf

from PIL import Image
from io import BytesIO
import glob
import cv2
import logging
import pandas as pd

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v2

# Patch the location of gfile
tf.gfile = tf.io.gfile

from di_log import p

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: a file path (this can be local or on colossus)
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """

    p('path: {}'.format(path))
    try:
        img_data = tf.io.gfile.GFile(path, 'rb').read()
    except AttributeError as error:
        p('error #113: {}'.format(error))

    p('CCC')
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    p('rifsi_00')
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    p('rifsi_01A')
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    p('rifsi_01B')

    # Run inference
    output_dict = model(input_tensor)
    p('rifsi_02')

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    p('rifsi_03')
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    p('rifsi_04')
    output_dict['num_detections'] = num_detections
    p('rifsi_05')

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    p('rifsi_06')

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        p('rifsi_07')
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        p('rifsi_08')
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        p('rifsi_09')
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        p('rifsi_10')

    p('rifsi_11')

    return output_dict

def run_inference_for_single_file(model, fn):
    p('=====================================')
    colorlist = [(0xff, 0x00, 0x00),(0x00, 0x80, 0x00),(0x00, 0x00, 0xff),(0x00, 0xff, 0xff)]
    df = pd.DataFrame(columns=['FileNameOd', 'Object', 'Prob', 'rRectXi', 'rRectYi', 'rRectXf', 'rRectYf', 'rCenterX', 'rCenterY', 'pxCenterX', 'pxCenterY', 'FileNameOdFull'])
    p('load_image_into_numpy_array')
    image_np = load_image_into_numpy_array(fn)
    # Actual detection.
    p('run_inference_for_single_image')
    output_dict = run_inference_for_single_image(model, image_np)

    p('cv2.imread')
    img = cv2.imread(fn, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    h, w = img.shape[:2]
    p('for det_prob, det_class, ')
    for det_prob, det_class, det_box in zip(output_dict['detection_scores'], output_dict['detection_classes'], output_dict['detection_boxes']):
        row = {}
        row['FileNameOd'] = os.path.basename(fn)
        row['Object'] = det_class
        row['Prob'] = det_prob
        row['rRectXi'] = det_box[1]
        row['rRectYi'] = det_box[0]
        row['rRectXf'] = det_box[3]
        row['rRectYf'] = det_box[2]
        row['rCenterX'] = (row['rRectXi'] + row['rRectXf']) / 2
        row['rCenterY'] = (row['rRectYi'] + row['rRectYf']) / 2
        row['pxCenterX'] = w * row['rCenterX']
        row['pxCenterY'] = h * row['rCenterY']
        row['FileNameOdFull'] = fn
        df = df.append(row, ignore_index=True)
        rw = (row['rRectXf'] - row['rRectXi']) * w
        rh = (row['rRectYf'] - row['rRectYi']) * h
        rect = [int(row['rRectXi']*w), int(row['rRectYi']*h), int(rw), int(rh)]
        color = (0 ,0 ,0)
        if det_class <= len(colorlist):
            color = colorlist[det_class - 1]
        if det_prob > 0.5:             
            cv2.rectangle(img, rect, color, thickness = 3)
            print(rect)
    return df, img



# %%

