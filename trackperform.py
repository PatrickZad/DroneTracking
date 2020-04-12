import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import cv2 as cv
from common import *

from tracking.yolo import YOLO
from tracking.deep_sort import preprocessing
from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker
from tracking.tools import generate_detections as gdet
from tracking.deep_sort.detection import Detection as ddet
from PIL import Image

'''
python==3.6
tensorflow==1.4.0 or tensorflow-gpu==1.4.0 with cuda8.0 and cudnn6.0
opencv-contrib-python==3.4.2.17
keras=2.1.5
'''
val_sequences_dir = os.path.join(visdrone_dataset_dir, 'VisDrone2019-VID_MOT-val', 'sequences')


class SequenceReader:
    def __init__(self, dir):
        self.__dir = dir
        # self.__framefiles = os.listdir(dir)
        self.__next = 1
        self.__size = None
        self.__length = len(os.listdir(dir))

    def __iter__(self):
        return self

    def __next__(self):
        if self.__next == self.__length:
            raise StopIteration
        filename = self.__filename(self.__next)
        filepath = os.path.join(self.__dir, filename)
        img_array = cv.imread(filepath)
        self.__next += 1
        # img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)  # rgb image
        # img_array = np.transpose(img_array, [2, 0, 1])
        return img_array, filename

    def frame_size(self):
        if self.__size is None:
            shape = cv.imread(os.path.join(self.__dir, '0000001.jpg')).shape
            self.__size = (shape[1], shape[0])  # (width,height)
        return self.__size

    @staticmethod
    def __filename(index):
        str_num = str(index)
        while len(str_num) < 7:
            str_num = '0' + str_num
        return str_num + '.jpg'


def track_perform(model_type='onseq', write_video=True):
    type_candidates = ['ondet', 'onseq', 'refine']
    assert model_type in type_candidates
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    if model_type == type_candidates[0]:
        yolo_file = os.path.join(model_base, 'ondet_trained_weights_final.h5')
        class_file = os.path.join(model_base, 'visdrone_classes11.txt')
    elif model_type == type_candidates[1]:
        yolo_file = os.path.join(model_base, 'onseq_trained_weights_final.h5')
        class_file = os.path.join(model_base, 'visdrone_classes.txt')
    else:
        yolo_file = os.path.join(model_base, 'refine_trained_weights_final.h5')
        class_file = os.path.join(model_base, 'visdrone_classes6.txt')
    anchor_file = os.path.join(model_base, 'yolo_anchors.txt')
    model_filename = os.path.join(model_base, 'mars-small128.pb')
    apperance_model = gdet.create_box_encoder(model_filename, batch_size=1)
    detector = YOLO(yolo_file, anchor_file, class_file, is_weights=True)
    apperance_metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    dirs = os.listdir(val_sequences_dir)
    for seq_dir in dirs[::-1]:
        tracker = Tracker(apperance_metric)
        frames_dir = os.path.join(val_sequences_dir, seq_dir)
        frame_reader = SequenceReader(frames_dir)

        if write_video:
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            writer = cv.VideoWriter(os.path.join(expr_dir, seq_dir + 'tracking.avi'), fourcc, 15,
                                    frame_reader.frame_size())

        frame_index = 0
        for (frame, filename) in frame_reader:
            frame = cv.imread(os.path.join(frames_dir, filename))
            img = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            raw_detections, raw_scores, raw_classifications = detector.detect_image(img)
            class_array = np.array(raw_classifications)
            det_array, score_array, class_array = np.array(raw_detections), np.array(raw_scores), np.array(
                raw_classifications)
            if model_type == type_candidates[0]:
                class_mask = class_array != 10

            elif model_type == type_candidates[1]:
                class_mask = np.logical_and(class_array != 0, class_array != 11)
            else:
                class_mask = class_array != 5
            valid_dets, scores, classes = det_array[class_mask], score_array[class_mask], class_array[class_mask]
            apperance_features = apperance_model(frame, valid_dets)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(valid_dets, apperance_features)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # track
            tracker.predict()
            tracker.update(detections)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

            for det, cla in zip(detections, raw_classifications):
                bbox = det.to_tlbr()
                cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            # cv.imwrite(os.path.join(expr_dir, filename), frame)
            # cv.imshow('', frame)
            if write_video:
                writer.write(frame)
            frame_index += 1
            print('frame ' + filename + ' complete !')
        if write_video:
            writer.release()


if __name__ == '__main__':
    track_perform('refine')
