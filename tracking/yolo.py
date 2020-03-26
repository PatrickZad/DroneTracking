#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from tracking.yolo3.model import yolo_eval, yolo_body
from tracking.yolo3.utils import letterbox_image


class YOLO:
    def __init__(self, model_path, anchor_path, classes_path, is_weights=False):
        self.model_path = model_path
        self.anchors_path = anchor_path
        self.classes_path = classes_path
        self.score = 0.5
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        # self.model_image_size = (416, 416)  # fixed size or (None, None)
        self.model_image_size = (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        with open(classes_path, 'r') as file:
            lines = file.readlines()
            num_class = len(lines)
        self.boxes, self.scores, self.classes = self.generate(is_weights, num_class)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self, is_weights, num_class):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        if not is_weights:
            self.yolo_model = load_model(model_path, compile=False)
        else:
            input_layer = Input(shape=(None, None, 3))
            self.yolo_model = yolo_body(input_layer, 3, num_class)
            self.yolo_model.load_weights(model_path, by_name=True, skip_mismatch=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        # raw_output = self.yolo_model.predict(image_data)
        # output = yolo_eval(raw_output, self.anchors, 11, [image.size[1], image.size[0]])
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        return_boxs = []
        return_classes = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            # if predicted_class != 'person' :
            #     continue
            box = out_boxes[i]
            # score = out_scores[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            # correct bbox
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxs.append([x, y, w, h])
            return_classes.append(c)

        return return_boxs, return_classes

    def close_session(self):
        self.sess.close()

    def __parse_raw_out(self, raw_out, anchors, num_classes, image_shape, max_boxes=20,
                        score_threshold=.6, iou_threshold=.5):
        '''Equal to yolo_eval'''
        anchor_group = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = raw_out[0].shape[1:3] * 32
        boxes = []
        box_scores = []
        for l in range(3):
            _boxes, _box_scores = self.__boxes_and_scores(raw_out[l], anchors[anchor_group[l]],
                                                          num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = np.concatenate(boxes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)
        objectness_mask = box_scores >= score_threshold
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            class_boxes = box_scores[objectness_mask[:, c]]
            class_box_scores = box_scores[:, c][objectness_mask[:, c]]

    def __boxes_and_scores(self, output, anchors, num_classes, input_shape, image_shape):
        '''Equal to yolo_boxes_and_scores'''
        box_xy, box_wh, box_confidence, box_class_probs = self.__split_out(output, anchors, num_classes, input_shape)
        boxes = self.__correct_boxes(box_xy, box_wh, input_shape, image_shape)  # min-max boxes
        boxes = boxes.reshape([-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = box_scores.reshape([-1, num_classes])
        return boxes, box_scores

    def __split_out(self, output, anchors, num_classes, input_shape):
        '''Equal to yolo_head'''
        num_achors = len(anchors)
        anchor_array = np.array(anchors).reshape((1, 1, 1, num_achors, 2))
        grid_shape = output.shape[1:3]
        grid_y = np.tile(np.arange(0, stop=grid_shape[0]).reshape(-1, 1, 1, 1), [1, grid_shape[1], 1, 1])
        grid_x = np.tile(np.arange(0, stop=grid_shape[1]).reshape([1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y])
        grid = np.float32(grid)
        ordered_output = output.reshape([-1, grid_shape[0], grid_shape[1], num_achors, num_classes + 5])
        box_xy = 1 / (1 + np.exp(-ordered_output[..., :2]))
        box_wh = np.exp(ordered_output[..., 2:4])
        box_confidence = 1 / (1 + np.exp(output[..., 4:5]))
        box_class_probs = 1 / (1 + np.exp(output[..., 5:]))

        box_xy = (box_xy + grid) / np.float32(grid_shape[::-1])
        box_wh = box_wh * anchor_array / np.float32(input_shape[::-1])
        return box_xy, box_wh, box_confidence, box_class_probs

    def __correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        '''Equal to yolo_correct_boxes'''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.float32(input_shape)
        image_shape = np.float32(image_shape)
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape  # deal with random scale in data augmentation
        box_yx = (box_yx - offset) * scale  # correct to input_shape scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        # TODO chang max-mins to cornor-size
        boxes = np.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes


def non_max_suppression_fast(boxes, iou_threshold):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > iou_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
