import numpy as np


# detection evaluation

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.zeros(inter_rect_x1.shape[0])
    raw_inter_width = inter_rect_x2 - inter_rect_x1 + 1
    raw_inter_height = inter_rect_y2 - inter_rect_y1 + 1
    mask = np.logical_and(raw_inter_width > 0, raw_inter_height > 0)
    inter_area[mask] = raw_inter_height[mask] * raw_inter_width[mask]
    '''inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )'''
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_batch_statistics(outputs, targets, iou_threshold):
    '''
    Compute true positives, predicted scores and predicted labels per sample.
    Parameters
    ----------
    outputs: a list with length batch_size,each element contains several
        [x1,y1,x2,y2,score,class]-detections for one image
    targets: a list with length batch_size,each element contains several
        [class,x1,y1,x2,y2]-annotations for one image
    iou_threshold

    Returns
    -------
    batch_metrics: list of batch_size [true_positives, pred_scores, pred_labels] lists,
        true_positives indicates a detection to be Tp or not with 1 or 0.
    '''

    batch_metrics = []
    for sample_i in range(len(outputs)):

        if len(outputs[sample_i]) == 0:  # is None:
            continue

        output = outputs[sample_i]  # output of one sample
        pred_boxes = output[:, :4]  # (x1,y1,x2,y2)
        pred_scores = output[:, 4]  # conf
        pred_labels = output[:, -1]  # pred label

        true_positives = np.zeros(pred_boxes.shape[0])  # number of boxes

        annotations = targets[sample_i]  # ground truth annotation: label,x1,y1,x2,y2
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]  # ground truth coordinates: x1,y1,x2,y2

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                iou_array = bbox_iou(np.tile(pred_box, (target_boxes.shape[0], 1)), target_boxes)
                box_index = np.argmax(iou_array)
                iou = iou_array[box_index]
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)  # i indicates max
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]  # sorted by confidence

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, precision, recall = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # number of class c in Ground Truth
        n_p = i.sum()  # number of class c in precision results

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            recall.append(0)
            precision.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()  # false positive
            tpc = (tp[i]).cumsum()  # true positive

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            recall.append(recall_curve[-1])  # recall of class c

            # Precision
            precision_curve = tpc / (tpc + fpc)
            precision.append(precision_curve[-1])  # precision of class c

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    precision, recall, ap = np.array(precision), np.array(recall), np.array(ap)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return precision, recall, ap, f1, unique_classes.astype("int32")


def detection_eval(raw_detections, raw_scores, raw_classifications, annotations, iou_thres):
    '''
    Parameters
    ----------
    raw_detections: a list of images' [x,y,w,h] bboxes
    raw_scores: a list of images' bbox scores
    raw_classifications: a list of images' bbox classifications
    annotations: a list of images' [x1,y1,x2,y2,class] annotations
    iou_thres
    conf_thres

    Returns
    -------

    '''
    img_nums = len(raw_detections)
    sample_metrics = []
    labels = []
    for i in range(img_nums):
        detboxes = np.array(raw_detections[i]).reshape((-1, 4))
        detboxes[:, 2] += detboxes[:, 0]
        detboxes[:, 3] += detboxes[:, 1]
        scores = np.array(raw_scores[i]).reshape((-1, 1))
        classifications = np.array(raw_classifications[i]).reshape((-1, 1))
        labels += annotations[i][:, -1].tolist()
        classes = annotations[i][:, -1].reshape((-1, 1))
        boxes = annotations[i][:, :-1].reshape((-1, 4))
        targets = np.concatenate([classes, boxes], axis=-1)
        targets = np.expand_dims(targets, 0)
        detections = np.concatenate([detboxes, scores, classifications], axis=-1)
        detections = np.expand_dims(detections, 0)
        # batch_size, each sample :[true_positives(list), pred_scores(list), pred_labels(list)]
        sample_metrics += get_batch_statistics(detections, targets, iou_threshold=iou_thres)
    # three lists: true_positives, pred_scores, pred_labels
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    # the format of callback is "vector" corresponding to the class order in ap_class
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    result_dict = {}
    result_dict['precision'] = precision.mean()
    result_dict['recall'] = recall.mean()
    result_dict['mAP'] = AP.mean()
    result_dict['f1'] = f1.mean()
    result_dict['ap_class'] = []
    for i, c in enumerate(ap_class):
        result_dict['ap_class'].append((c, AP[i]))
    return result_dict


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
