import os
import numpy as np
import cv2 as cv

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
expr_base = '/home/patrick/PatrickWorkspace/DroneTracking/experiments'
visdrone = '/home/patrick/PatrickWorkspace/Datasets/VisDrone'
val_sequences_dir = os.path.join(visdrone, 'VisDrone2019-VID_MOT-val', 'sequences')


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

    def __filename(self, index):
        str_num = str(index)
        while len(str_num) < 7:
            str_num = '0' + str_num
        return str_num + '.jpg'


write_video = True

# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort
model_base = '/home/patrick/PatrickWorkspace/DroneTracking/tracking/model_data'
# yolo_file = os.path.join(model_base, 'trained_weights_final.h5')
yolo_file = os.path.join(model_base, 'yolo.h5')
anchor_file = os.path.join(model_base, 'yolo_anchors.txt')
# class_file = os.path.join(model_base, 'visdrone_classes.txt')
class_file = os.path.join(model_base, 'coco_classes.txt')
model_filename = os.path.join(model_base, 'mars-small128.pb')
apperance_model = gdet.create_box_encoder(model_filename, batch_size=1)
detector = YOLO(yolo_file, anchor_file, class_file, is_weights=True)
apperance_metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(apperance_metric)
dirs = os.listdir(val_sequences_dir)
for dir in dirs:
    frames_dir = '/home/patrick/PatrickWorkspace/Datasets/rocks'
    # frames_dir = os.path.join(val_sequences_dir, dir)
    # frame_reader = SequenceReader(frames_dir)
    '''
    if write_video:
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        writer = cv.VideoWriter(os.path.join(expr_base, dir + 'tracking.avi'), fourcc, 15, frame_reader.frame_size())
    '''
    frame_index = 0
    # for (frame, filename) in frame_reader:
    for filename in os.listdir(frames_dir):
        frame = cv.imread(os.path.join(frames_dir, filename))
        img = Image.fromarray(frame)
        raw_detections, raw_classifications = detector.detect_image(img)
        apperance_features = apperance_model(frame, raw_detections)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(raw_detections, apperance_features)]
        # Run non-maxima suppression.
        '''boxes = np.array([d.tlwh for d in detections])
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
        '''
        for det, cla in zip(detections, raw_classifications):
            bbox = det.to_tlbr()
            cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv.putText(frame, str(cla), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (255, 0, 0), 2)
        cv.imwrite(os.path.join(expr_base, filename), frame)
        # cv.imshow('', frame)
        # if write_video:
        #     writer.write(frame)
        frame_index += 1
        print('frame ' + filename + ' complete !')
    break
    # if write_video:
    #     writer.release()
    # cv.destroyAllWindows()
