import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
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
from evaluate import bbox_iou

'''
python==3.6
tensorflow==1.4.0 or tensorflow-gpu==1.4.0 with cuda8.0 and cudnn6.0
opencv-contrib-python==3.4.2.17
keras=2.1.5
'''
val_sequences_dir = os.path.join(visdrone_dataset_dir, 'VisDrone2019-VID_MOT-val', 'sequences')
det_eval_types = ['dynamic', 'static']
mot_eval_types = ['dynamic', 'tiny_occlusion_static', 'heavy_occlusion_static', 'no_occlusion_static']
refined_valid_categories = [1, 2, 4, 5, 6, 9]


class SeqPerformEvaluator:
    def __init__(self, anno_file):
        self.__targets_on_id = {}
        self.__targets_on_frame = {}
        with open(anno_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                nums = np.array(line[:-1].split(','))
                if nums[7] in refined_valid_categories:
                    target_id = nums[1]
                    frame_id = nums[0]
                    if frame_id not in self.__targets_on_frame.keys():
                        self.__targets_on_frame[frame_id] = []
                    if target_id in self.__targets_on_id.keys():
                        target = self.__targets_on_id[target]
                    else:
                        target = DT_Target(target)
                        self.__targets_on_id[target_id] = target
                    self.__targets_on_frame[frame_id].append(target)
                    target.add_info(frame_id, nums[2:6], nums[7], nums[8], nums[9])

    def get_target(self, id):
        assert id in self.__targets.keys()
        return self.__targets_on_id[id]

    def get_frame_targets(self, frame_id):
        return self.__targets_on_frame[frame_id]

    def update_det(self, frame_id, detections):
        taregts_in_frame = self.get_frame_targets(frame_id)
        target_boxes = np.array([target.get_info()['bbox'] for target in taregts_in_frame])
        for bbox in detections:
            box_array = np.array(bbox)
            box_array[:, 2:] += box_array[:, :2]  # to min-max
            ious = bbox_iou(box_array, target_boxes)
            target_index = np.argmax(ious)
            taregts_in_frame[target_index].add_det_mark(frame_id, True)
        for target in taregts_in_frame:
            target.set_det_default()

    def update_mot(self, frame_id, track):
        taregts_in_frame = self.get_frame_targets(frame_id)
        target_boxes = np.array([target.get_info()['bbox'] for target in taregts_in_frame])
        track_box = track.to_tlbr()
        ious = bbox_iou(track_box, target_boxes)
        target_index = np.argmax(ious)
        taregts_in_frame[target_index].add_track_id(frame_id, track.track_id)

    def eval(self):
        # det eval,all targets, dynamic targets, static targets
        all_targets_num = 0
        detected_targets_num = 0
        all_static_nums = 0
        detected_static_nums = 0
        for frame_id, targets in self.__targets_on_frame.items():
            all_targets_num += len(targets)
            for target in targets:
                if target.eval_det_type[frame_id] == det_eval_types[1]:
                    all_static_nums += 1
                if target.det_marks[frame_id]:
                    detected_targets_num += 1
                    if target.eval_det_type[frame_id] == det_eval_types[1]:
                        detected_static_nums += 1
        self.print_calculation('All det precision', detected_targets_num, all_targets_num)
        self.print_calculation('Dynamic det precision', detected_targets_num - detected_static_nums,
                               all_targets_num - all_static_nums)
        self.print_calculation('Static det precision', detected_static_nums, all_static_nums)
        # mot eval, mean association precision,static association precision,
        # dynamic association precision,occlusion association precision
        asso_target_nums = 0
        asso_static_target_nums = 0
        asso_dynamic_target_nums = 0
        asso_occlusion_target_nums = 0
        asso_correct_nums = 0
        asso_static_correct_nums = 0
        asso_dynamic_correct_nums = 0
        asso_occlusion_correct_nums = 0
        frame_ids = self.__targets_on_frame.keys()
        for frame_index in range(len(frame_ids) - 1):
            next_frame_id = frame_ids[frame_index + 1]
            current_frame_id = frame_ids[frame_index]
            next_frame_targets = self.__targets_on_frame[next_frame_id]
            asso_target_nums += len(next_frame_targets)
            for target in next_frame_targets:
                correct_asso = current_frame_id in target.track_ids.keys() and target.track_ids[current_frame_id] == \
                               target.track_ids[next_frame_id]
                if correct_asso: asso_correct_nums += 1
                if target.eval_mot_type[next_frame_id] == mot_eval_types[0]:
                    asso_dynamic_target_nums += 1
                    if correct_asso: asso_dynamic_correct_nums += 1
                else:
                    asso_static_target_nums == 1
                    if correct_asso: asso_static_correct_nums += 1
                    if target.eval_mot_type[next_frame_id] == mot_eval_types[1]:
                        asso_occlusion_target_nums += 1
                        if correct_asso:
                            asso_occlusion_correct_nums += 1
        self.print_calculation('All MOT precision', asso_correct_nums, asso_target_nums)
        self.print_calculation('Dynamic MOT precision', asso_dynamic_correct_nums, asso_dynamic_target_nums)
        self.print_calculation('Static MOT precision', asso_static_correct_nums, asso_static_target_nums)
        self.print_calculation('Occlusion MOT precision', asso_occlusion_correct_nums, asso_occlusion_target_nums)

    @staticmethod
    def print_calculation(msg, num, den):
        print(msg + ' : ' + str(num) + '/' + str(den) + '=' + str(num / den))


class DT_Target:
    def __init__(self, id):
        self.id = id
        self.info_at_frame = {}
        self.track_ids = {}
        self.det_marks = {}
        self.eval_det_type = {}
        self.eval_mot_type = {}

    def add_info(self, frame_id, bbox, category, truncation, occlusion):
        # change to min-max bbox
        min_max_box = np.array([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]])
        if category == 2:
            self.eval_det_type[frame_id] = det_eval_types[1]
            if truncation + occlusion >= 2:  # one of both is 2 or both are 1
                self.eval_mot_type[frame_id] = mot_eval_types[2]
            elif truncation == 0 and occlusion == 0:
                self.eval_mot_type[frame_id] = mot_eval_types[3]
            else:
                self.eval_mot_type[frame_id] = mot_eval_types[1]
        else:
            self.eval_det_type[frame_id] = det_eval_types[0]
            self.eval_mot_type[frame_id] = mot_eval_types[0]
        self.info_at_frame[frame_id] = {'bbox': min_max_box, 'truncation': truncation, 'occlusion': occlusion}

    def get_info(self, frame_id):
        return self.info_at_frame[frame_id]

    def add_track_id(self, frame_id, track_id):
        self.track_ids[frame_id] = track_id

    def add_det_mark(self, frame_id, isdetected):
        self.det_marks[frame_id] = isdetected

    def set_det_default(self, frame_id, default=False):
        if frame_id not in self.det_marks.keys():
            self.det_marks[frame_id] = default


class SequenceReader:
    def __init__(self, dir):
        self.__dir = dir
        # self.__framefiles = os.listdir(dir)
        self.__next = 1  # frame index
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
        return img_array, self.__next - 1, filename

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
        anno_file_path = os.path.join(mot_val_anno_dir, seq_dir + '.txt')

        evaluator = SeqPerformEvaluator(anno_file_path)

        if write_video:
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            writer = cv.VideoWriter(os.path.join(expr_dir, seq_dir + 'tracking.avi'), fourcc, 15,
                                    frame_reader.frame_size())

        frame_index = 0
        for (frame, frame_id, filename) in frame_reader:
            frame = cv.imread(os.path.join(frames_dir, filename))
            img = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            raw_detections, raw_scores, raw_classifications = detector.detect_image(img)

            evaluator.update_det(frame_id, raw_detections)

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
            apperance_features = apperance_model(frame, detections)
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

                evaluator.update_mot(frame_id, track)

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
