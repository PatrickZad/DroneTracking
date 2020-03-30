import numpy as np
import cv2 as cv
from common import *
from PIL import Image
from tracking.yolo import YOLO
from evaluate import detection_eval


class DetImageReader:
    def __init__(self, base_dir):
        self.__img_dir = os.path.join(base_dir, 'images')
        self.__anno_dir = os.path.join(base_dir, 'annotations')
        self.__fileids = [filename[:-4] for filename in os.listdir(self.__img_dir)][:2]
        self.__next = 0
        # self.__length = len(self.__fileids)
        self.__length = 2

    def __iter__(self):
        return self

    def __next__(self):
        if self.__next == self.__length:
            raise StopIteration
        imgpath = os.path.join(self.__img_dir, self.__fileids[self.__next] + '.jpg')
        annopath = os.path.join(self.__anno_dir, self.__fileids[self.__next] + '.txt')
        self.__next += 1
        img_array = cv.imread(imgpath)
        anno_list = []
        with open(annopath, 'r') as anno_file:
            lines = anno_file.readlines()
            for line in lines:
                nums = line.split(',')[:6]
                nums = np.int32(nums)
                if nums[4] != 1 or nums[5] == 0:
                    continue
                # to [x1,y1,x2,y2,class]
                anno_array = np.array([nums[0], nums[1], nums[0] + nums[2], nums[1] + nums[3],
                                       nums[5] - 1])  # for det trained model,class_id-1
                anno_array = np.expand_dims(anno_array, 0)
                anno_list.append(anno_array)
        # img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)  # rgb image
        # img_array = np.transpose(img_array, [2, 0, 1])
        return self.__fileids[self.__next-1], img_array, np.concatenate(anno_list, axis=0)


# yolo_file = os.path.join(model_base, 'trained_weights_stage_1.h5')
yolo_file = os.path.join(model_base, 'ondet_trained_weights_final.h5')
# yolo_file = os.path.join(model_base, 'yolo.h5')
anchor_file = os.path.join(model_base, 'yolo_anchors.txt')
class_file = os.path.join(model_base, 'visdrone_classes11.txt')
detector = YOLO(yolo_file, anchor_file, class_file, is_weights=True)
imgreader = DetImageReader(det_val_data_dir)
val_bboxes = []
val_scores = []
val_classes = []
val_annos = []
expr_base = os.path.join(expr_dir, 'det_val')
for file_id, img_array, annotation in imgreader:
    img = Image.fromarray(img_array[..., ::-1])  # bgr to rgb
    raw_detections, raw_scores, raw_classifications = detector.detect_image(img)
    for anno in annotation:
        cv.rectangle(img_array, (anno[0], anno[1]), (anno[2], anno[3]), (255, 255, 255), 2)
        cv.putText(img_array, str(anno[4]), (anno[2], anno[1]), 0, 5e-3 * 200, (0, 0, 255))
    for i in range(len(raw_detections)):
        minmax_bbox = (raw_detections[i][0], raw_detections[i][1], raw_detections[i][0] + raw_detections[i][2],
                       raw_detections[i][1] + raw_detections[i][3])
        cv.rectangle(img_array, (minmax_bbox[0], minmax_bbox[1]), (minmax_bbox[2], minmax_bbox[3]), (255, 0, 0), 2)
        cv.putText(img_array, str(raw_classifications[i]), (minmax_bbox[0], minmax_bbox[1]), 0, 5e-3 * 200, (0, 255, 0))

    cv.imwrite(os.path.join(expr_base, file_id + '.jpg'), img_array)
    print(file_id + ' detection complete !')
    val_bboxes.append(raw_detections)
    val_scores.append(raw_scores)
    val_classes.append(raw_classifications)
    val_annos.append(annotation)
evaluation = detection_eval(val_bboxes, val_scores, val_classes, val_annos, 0.5)
print(evaluation)
