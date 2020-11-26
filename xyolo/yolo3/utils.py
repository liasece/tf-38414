"""Miscellaneous utility functions."""

from functools import reduce
from operator import pos

import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import tensorflow as tf

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def get_random_data_xyolo(image_decoded, box, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    # print('get_random_data_xyolo')
    '''random preprocessing for real-time data augmentation'''
    image_decoded = image_decoded.numpy().astype(np.int8)
    box = box.numpy()
    input_shape = input_shape.numpy()
    image = Image.fromarray(image_decoded, "RGB")
    iw, ih = image.size
    h, w = input_shape
    # image.show()

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    # print(err)
    return image_data, box_data

def get_random_data_xyolo_warp(example_proto, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "height": tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        "width": tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        "channels": tf.io.FixedLenFeature([1], tf.int64, default_value=[3]),
        "colorspace": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "img_format": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.VarLenFeature(tf.int64),
        "bbox_xmin": tf.io.VarLenFeature(tf.int64),
        "bbox_xmax": tf.io.VarLenFeature(tf.int64),
        "bbox_ymin": tf.io.VarLenFeature(tf.int64),
        "bbox_ymax": tf.io.VarLenFeature(tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string, default_value="")
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    label = tf.expand_dims(parsed_features["label"].values, 0)
    label = tf.cast(label, tf.float32)
    image_raw  = tf.image.decode_jpeg(parsed_features["image"], channels=3)
    image_decoded = tf.cast(image_raw, dtype=tf.float32)
    filename = parsed_features["filename"]
    #Get the coco image id as we need to use COCO API to evaluate
    image_id = tf.strings.to_number(tf.strings.substr(filename, -16, 12), tf.int32)
    image_id = tf.expand_dims(image_id, 0)
    #Get the bbox
    xmin = tf.cast(tf.expand_dims(parsed_features["bbox_xmin"].values, 0), tf.float32)
    xmax = tf.cast(tf.expand_dims(parsed_features["bbox_xmax"].values, 0), tf.float32)
    ymin = tf.cast(tf.expand_dims(parsed_features["bbox_ymin"].values, 0), tf.float32)
    ymax = tf.cast(tf.expand_dims(parsed_features["bbox_ymax"].values, 0), tf.float32)
    boxes = tf.concat([xmin,ymin,xmax,ymax,label], axis=0)
    boxes = tf.transpose(boxes, [1, 0])
    # features = {'images':image_decoded, 'bbox':boxes, 'image_id':image_id}
    # y_true_0, y_true_1, y_true_2 = tf.py_function(preprocess_true_boxes_py, inp=[rand_box_data, input_shape, anchors, num_classes], Tout=[tf.float32,tf.float32,tf.float32])
    # features = {'images':rand_image_data, 'y_true_0':y_true_0, 'y_true_1':y_true_1, 'y_true_2':y_true_2}
    # print("parse_batch_function: ", features)
    # print(err)
    image,box = tf.py_function(get_random_data_xyolo, inp=[image_decoded, boxes, input_shape, random, max_boxes, jitter, hue, sat, val, proc_img], Tout=[tf.float32,tf.float32])
    features = {'images':image, 'bbox':box}
    return features


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def parse_function(example_proto, input_shape, max_boxes):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "height": tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        "width": tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        "channels": tf.io.FixedLenFeature([1], tf.int64, default_value=[3]),
        "colorspace": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "img_format": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.VarLenFeature(tf.int64),
        "bbox_xmin": tf.io.VarLenFeature(tf.int64),
        "bbox_xmax": tf.io.VarLenFeature(tf.int64),
        "bbox_ymin": tf.io.VarLenFeature(tf.int64),
        "bbox_ymax": tf.io.VarLenFeature(tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string, default_value="")
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    label = tf.expand_dims(parsed_features["label"].values, 0)
    label = tf.cast(label, tf.float32)
    # filename = parsed_features["filename"]
    #Get the coco image id as we need to use COCO API to evaluate
    # image_id = tf.strings.to_number(tf.strings.substr(filename, -16, 12), tf.int32)
    # image_id = tf.expand_dims(image_id, 0)

    #Get the bbox
    # proc image
    width, height = parsed_features["width"],parsed_features["height"]
    h, w = input_shape
    scale = w / width
    if h / height < scale:
        scale = h / height
    scale = tf.cast(scale, tf.float64)
    nw = tf.cast(width, tf.float64) * scale
    nh = tf.cast(height, tf.float64) * scale
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    xmin = tf.cast(tf.cast(tf.expand_dims(parsed_features["bbox_xmin"].values, 0), tf.float64) * scale + dx, tf.float32)
    xmax = tf.cast(tf.cast(tf.expand_dims(parsed_features["bbox_xmax"].values, 0), tf.float64) * scale + dx, tf.float32)
    ymin = tf.cast(tf.cast(tf.expand_dims(parsed_features["bbox_ymin"].values, 0), tf.float64) * scale + dy, tf.float32)
    ymax = tf.cast(tf.cast(tf.expand_dims(parsed_features["bbox_ymax"].values, 0), tf.float64) * scale + dy, tf.float32)
    boxes = tf.concat([xmin,ymin,xmax,ymax,label], axis=0)
    boxes = tf.transpose(boxes, [1, 0])
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    boxes = boxes[tf.logical_and(box_w > 1, box_h > 1)]
    if len(boxes) > max_boxes: boxes = boxes[:max_boxes]
    boxes = tf.pad(boxes, [[0,max_boxes - tf.shape(boxes)[0]],[0,0]],"CONSTANT")
    # box normalization
    # input_shape_tensor = tf.cast(input_shape, tf.float64)
    # box1_x = ((xmin + xmax) // 2) // input_shape[1]
    # box1_y = ((ymin + ymax) // 2) // input_shape[0]
    # box1_w = (xmax - xmin) // input_shape[1]
    # box1_h = (ymax - ymin) // input_shape[0]
    # box1 = tf.concat([box1_x,box1_y,box1_w,box1_h,label], axis=0)
    # box1 = tf.transpose(box1, [1, 0])
    # box1 = box1[tf.logical_and(box_w > 1, box_h > 1)]
    # if len(box1) > max_boxes: box1 = box1[:max_boxes]
    # box1 = tf.pad(box1, [[0,max_boxes - tf.shape(box1)[0]],[0,0]],"CONSTANT")

    # image
    image_decoded = tf.cast(tf.image.decode_jpeg(parsed_features["image"], channels=3), dtype=tf.float32)
    image = tf.image.resize(image_decoded, (w, h), ResizeMethod.BICUBIC)
    image = image / 255.

    # features = {'images':image, 'bbox':boxes, 'image_id':image_id, 'width': width, 'height': height, 'box1': box1}
    features = {'images':image, 'bbox':boxes}
    # print("parse_function: ", features)
    return features

def preprocess_true_boxes_py(true_boxes, input_shape, anchors, num_classes):
    true_boxes = true_boxes.numpy()
    input_shape = input_shape.numpy()
    anchors = anchors.numpy()
    # assert (true_boxes[..., 4] < num_classes).all(), 'class id{:} must be less than num_classes{:}'.format(true_boxes, num_classes)
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        # print('valid_mask[b]', valid_mask[b])
        # print('boxes_wh[b, valid_mask[b]]', boxes_wh[b, valid_mask[b]])
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        # print(" b: ", b, " best_anchor: ", len(best_anchor), " num_layers: ", num_layers)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    if len(y_true[l][b]) <= j :
                        print("out of bounds: j", len(y_true[l][b]), i)
                        continue
                    if len(y_true[l][b, j]) <= i :
                        print("out of bounds: i", len(y_true[l][b, j]), i)
                        continue
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
    # for l in range(num_layers):
    #     y_true[l] = y_true[l].tolist()
    return y_true[0], y_true[1], y_true[2]

def preprocess_true_boxes_wrap(example_proto, input_shape, batch_size, max_boxes, anchors, num_classes):
    # input_shape = np.array(input_shape_py, dtype='int32')
    # anchors = tf.cast(anchors, dtype='float32')
    # true_boxes = example_proto['bbox']
    rand_image_data,rand_box_data = example_proto['images'], example_proto['bbox']
    y_true_0, y_true_1, y_true_2 = tf.py_function(preprocess_true_boxes_py, inp=[rand_box_data, input_shape, anchors, num_classes], Tout=[tf.float32,tf.float32,tf.float32])
    features = {'images':rand_image_data, 'y_true_0':y_true_0, 'y_true_1':y_true_1, 'y_true_2':y_true_2}
    # print("parse_batch_function: ", features)
    # print(err)
    return features, tf.zeros(batch_size)

def load_tfrecord_dataset(dir, input_shape, batch_size, anchors, num_classes, max_boxes = 20):
    return load_tfrecord_dataset_xyolo(dir, input_shape, batch_size, anchors, num_classes, max_boxes)
    files = tf.data.Dataset.list_files(os.path.join(dir,"*.tfrecord"))
    dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    # parse tfrecord
    dataset = dataset.map(lambda x: parse_function(x, input_shape, max_boxes), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).prefetch(buffer_size = batch_size * 2)
    # parse batch
    dataset = dataset.batch(batch_size).prefetch(buffer_size = 2)
    # y-true
    dataset = dataset.map(lambda x: preprocess_true_boxes_wrap(x, input_shape, batch_size, max_boxes, anchors, num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    # prefetch
    dataset = dataset.prefetch(buffer_size = 8).apply(tf.data.experimental.copy_to_device("/gpu:0"))
    with tf.device("/gpu:0"):
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def preprocess_true_boxes_xyolo(true_boxes, input_shape, anchors, num_classes):
    true_boxes = true_boxes.numpy()
    input_shape = input_shape.numpy()
    anchors = anchors.numpy()
    # assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
    return y_true[0], y_true[1], y_true[2]

def preprocess_true_boxes_xyolo_wrap(example_proto, input_shape, batch_size, max_boxes, anchors, num_classes):
    rand_image_data,rand_box_data = example_proto['images'], example_proto['bbox']
    y_true_0, y_true_1, y_true_2 = tf.py_function(preprocess_true_boxes_xyolo, inp=[rand_box_data, input_shape, anchors, num_classes], Tout=[tf.float32,tf.float32,tf.float32])
    features = {'images':rand_image_data, 'y_true_0':y_true_0, 'y_true_1':y_true_1, 'y_true_2':y_true_2}
    return features, tf.zeros(batch_size)

def load_tfrecord_dataset_xyolo(dir, input_shape, batch_size, anchors, num_classes, max_boxes = 20):
    files = tf.data.Dataset.list_files(os.path.join(dir,"*.tfrecord"))
    dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    # parse tfrecord
    dataset = dataset.map(lambda x: parse_function(x, input_shape, max_boxes), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).repeat().shuffle(2)
    # parse batch
    dataset = dataset.batch(batch_size)
    # y-true
    dataset = dataset.map(lambda x: preprocess_true_boxes_xyolo_wrap(x, input_shape, batch_size, max_boxes, anchors, num_classes), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    # prefetch
    dataset = dataset.prefetch(buffer_size = 8).apply(tf.data.experimental.copy_to_device("/gpu:0"))
    with tf.device("/gpu:0"):
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
