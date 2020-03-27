import numpy as np


# detection evaluation
def average_precision(class_ids, detections, annotations, index='normal'):
    # detections and annotations are batches of group of iterable objects of tuple (x,y,w,h,scores,class)
    # assert index in ['normal', '50', '75', 'small', 'middle', 'large']
    ap_sum = 0
    for class_id in class_ids:
        all_detections = 0
        all_annotations = 0
        tp = 0
        for m in range(detections.shape[0]):
            class_detections = detections[m][detections[:, -1] == class_id]
            class_detections = class_detections[np.argsort(class_detections[:, -2])[::-1]]
            class_annotations = annotations[m][annotations[:, -1] == class_id]
            all_detections += class_detections.shape[0]
            all_annotations += class_annotations.shape[0]


# tracking evaluation
def mota():
    pass


def motp():
    pass


def mt():
    pass


def id_switch():
    pass


def fm():
    pass


def fp():
    pass


def fn():
    pass
