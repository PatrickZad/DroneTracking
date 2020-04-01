import cv2 as cv
import numpy as np
from common import *
from collections import defaultdict
import multiprocessing

from tracking.yolo3.model import preprocess_true_boxes
from tracking.yolo3.utils import get_random_data


def train_generator(batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    fileids = [filename[:-4] for filename in os.listdir(det_train_img_dir)]
    n = len(fileids)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(fileids)
            image, box = det_data_augment(det_train_img_dir, det_train_anno_dir, fileids[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def val_generator(batch_size, input_shape, anchors, num_classes):
    fileids = [filename[:-4] for filename in os.listdir(det_val_img_dir)]
    n = len(fileids)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(fileids)
            image, box = det_data_augment(det_val_img_dir, det_val_anno_dir, fileids[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def det_data_augment(img_dir, anno_dir, fileid, input_shape):
    img = cv.imread(os.path.join(img_dir, fileid + '.jpg'))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.float32(img)
    anno_array = parse_det_anno(anno_dir, fileid)
    anno_array = np.array(anno_array)
    image, box = get_random_data(img, anno_array, input_shape, random=True, max_boxes=128)
    return image, box


def seq_data_augment_for_det(fileid, input_shape, seq_annos, phase='train', refined_class=False):
    assert phase == 'train' or phase == 'val'
    if phase == 'train':
        seq_base = mot_train_seq_dir
    else:
        seq_base = mot_val_seq_dir
    img_name = str(fileid[1])
    while len(img_name) < 7:
        img_name = '0' + img_name
    img = cv.imread(os.path.join(seq_base, fileid[0], img_name + '.jpg'))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.float32(img)
    anno_array = parse_seq_anno_for_det(seq_annos, fileid, refined_class)
    image, box = get_random_data(img, anno_array, input_shape, random=True, max_boxes=128)
    return image, box


def parse_det_anno(anno_dir, fileid):
    result = []
    with open(os.path.join(anno_dir, fileid + '.txt')) as file:
        line = file.readline()
        while len(line) > 0:
            strnums = line.split(',')[:6]
            line = file.readline()
            nums = []
            for str_num in strnums:
                nums.append(int(str_num))
            if nums[4] != 1 or nums[5] == 0:
                continue
                # chang to min-max box
            nums[2] += nums[0]
            nums[3] += nums[1]
            nums[5] -= 1
            anno = nums[:4]
            anno.append(nums[5])
            result.append(anno)
    return result


def parse_seq_anno_for_det(anno_dict, file_id, refined_class):
    annos = anno_dict[file_id[0]][file_id[1]]
    anno_array = np.concatenate(annos, axis=0)
    anno_array[:, 2] += anno_array[:, 0]
    anno_array[:, 3] += anno_array[:, 1]
    if refined_class:
        # remove ignore regions
        mask = anno_array[:, 4] != 0
        anno_array = anno_array[mask]
        # merge people and pedestrian
        anno_array[:, 4][anno_array[:, 4] <= 2] = 0
        # id3 to be 11 to make sure no wrong change
        anno_array[:, 4][anno_array[:, 4] == 3] = 11
        # car,van,truck,bus
        anno_array[:, 4][anno_array[:, 4] == 4] = 1
        anno_array[:, 4][anno_array[:, 4] == 5] = 2
        anno_array[:, 4][anno_array[:, 4] == 6] = 3
        anno_array[:, 4][anno_array[:, 4] == 9] = 4
        # others
        anno_array[:, 4][anno_array[:, 4] > 4] = 5
    return anno_array

def all_seq_data_for_det(phase='train'):
    assert phase == 'train' or phase == 'val'
    if phase == 'train':
        seq_dir = mot_train_seq_dir
        anno_dir = mot_train_anno_dir
    else:
        anno_dir = mot_val_anno_dir
        seq_dir = mot_val_seq_dir
    videoids = [filename[:-4] for filename in os.listdir(anno_dir)]
    anno_dict = {}
    seq_ids=[]
    for vid in videoids:
        vanno_dict = {}
        with open(os.path.join(anno_dir, vid + '.txt'), 'r') as anno_file:
            lines = anno_file.readlines()
            for line in lines:
                nums = line.split(',')
                id_num = int(nums[0])
                anno_num = np.int32([nums[2], nums[3], nums[4], nums[5], nums[7]])
                if id_num in vanno_dict.keys():
                    vanno_dict[id_num].append(anno_num.reshape((1, 5)))
                else:
                    seq_ids.append((vid,id_num))
                    vanno_dict[id_num]=[anno_num.reshape((1, 5))]
        anno_dict[vid] = vanno_dict
    return seq_ids,anno_dict


def all_seq_ids(phase='train'):
    assert phase == 'train' or phase == 'val'
    if phase == 'train':
        seq_dir = mot_train_seq_dir
    else:
        seq_dir = mot_val_seq_dir
    result = []
    for vdir in os.listdir(seq_dir):
        for i in range(1, len(os.listdir(os.path.join(seq_dir, vdir))) + 1):
            result.append((vdir, i))
    return result


def all_seq_annos(phase='train'):
    # dict {videoid:{frameid:lines}}
    assert phase == 'train' or phase == 'val'
    if phase == 'train':
        anno_dir = mot_train_anno_dir
    else:
        anno_dir = mot_val_anno_dir
    videoids = [filename[:-4] for filename in os.listdir(anno_dir)]
    '''anno_dict = Manager().dict()

    def task(mgr_dict, vid):
        vanno_dict = defaultdict(list)
        with open(os.path.join(anno_dir, vid + '.txt'), 'r') as anno_file:
            lines = anno_file.readlines()
            i = 0
            for line in lines:
                i += 1
                nums = line.split(',')
                if nums[7] == '0' or nums[9] == '2':
                    continue
                id_num = int(nums[0])
                anno_num = np.int32([nums[2], nums[3], nums[4], nums[5], nums[7]])
                vanno_dict[id_num].append(anno_num)
        mgr_dict[vid] = vanno_dict
        print('load anno: ' + vid)

    cores = cpu_count() // 2
    proc_pool = Pool(cores)
    for vid in videoids:
        proc_pool.apply_async(task, args=(anno_dict, vid))
    proc_pool.close()
    proc_pool.join()'''
    anno_dict = {}
    for vid in videoids:
        vanno_dict = defaultdict(list)
        with open(os.path.join(anno_dir, vid + '.txt'), 'r') as anno_file:
            lines = anno_file.readlines()
            i = 0
            for line in lines:
                i += 1
                nums = line.split(',')
                id_num = int(nums[0])
                anno_num = np.int32([nums[2], nums[3], nums[4], nums[5], nums[7]])
                vanno_dict[id_num].append(anno_num.reshape((1, 5)))
        anno_dict[vid] = vanno_dict
        # print('load anno: ' + vid)
    return anno_dict



def seq_train_for_det_generator(batch_size, input_shape, anchors, num_classes, file_ref=None,refined_class=False):
    if file_ref is None:
        fileids,seq_annos = all_seq_data_for_det()
    else:
        fileids, seq_annos=file_ref[0],file_ref[1]
    n = len(fileids)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(fileids)
            image, box = seq_data_augment_for_det(fileids[i], input_shape, seq_annos, refined_class=refined_class)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def seq_val_for_det_generator(batch_size, input_shape, anchors, num_classes, file_ref=None,refined_class=False):
    if file_ref is None:
        fileids,seq_annos = all_seq_data_for_det('val')
    else:
        fileids, seq_annos=file_ref[0],file_ref[1]
    n = len(fileids)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(fileids)
            image, box = seq_data_augment_for_det(fileids[i], input_shape, seq_annos, 'val', refined_class)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def cluster_anchors(boxes, k, dist=np.median):
    box_number = boxes.shape[0]
    distances = np.empty((box_number, k))
    last_nearest = np.zeros((box_number,))
    np.random.seed()
    clusters = boxes[np.random.choice(
        box_number, k, replace=False)]  # init k clusters
    turn = 0
    while True:

        distances = 1 - iou4array(boxes, clusters, k)

        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        last_clusters = clusters.copy()
        for cluster in range(k):
            clusters[cluster] = dist(  # update clusters
                boxes[current_nearest == cluster], axis=0)
        if turn % 50 == 0:
            delta = clusters - last_clusters
            print('turn: ' + str(turn))
            print(delta)
        turn += 1
        last_nearest = current_nearest

    return clusters


def iou4array(boxes, cluster_centers, k):
    # return a n*k iou matrix
    n = boxes.shape[0]
    box_area = boxes[:, 2] * boxes[:, 3]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = cluster_centers[:, 2] * cluster_centers[:, 3]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape((boxes[:, 2] - boxes[:, 0]).repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(cluster_centers[:, 2], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 3].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(cluster_centers[:, 3], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area)
    return result


def det_statistics():
    count_array = np.zeros(12)
    for anno_file in os.listdir(det_train_anno_dir):
        with open(os.path.join(det_train_anno_dir, anno_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                class_num = int(line.split(',')[5])
                count_array[class_num] += 1
    return count_array


def count_task(count_array, count_file):
    with open(os.path.join(mot_train_anno_dir, count_file), 'r') as file:
        line = file.readline()
        while len(line) > 0:
            class_num = int(line.split(',')[7])
            count_array[class_num] += 1
            line = file.readline()


def seq_statistics():
    '''count_array = multiprocessing.Manager().list([0] * 12)
    p_pool = multiprocessing.Pool(4)
    for filename in os.listdir(mot_train_anno_dir):
        p_pool.apply_async(count_task, args=(count_array, filename))
    p_pool.close()
    p_pool.join()'''
    count_array = np.zeros(12)
    for filename in os.listdir(mot_train_anno_dir):
        count_task(count_array, filename)
    return count_array


if __name__ == '__main__':
    count = det_statistics()
    print(count)
    count = seq_statistics()
    print(count)
    '''boxes_dict = all_seq_annos()
    all_boxes = []
    for vid, frame_boxes_dict in boxes_dict.items():
        for id, boxes in frame_boxes_dict.items():
            array = np.concatenate(boxes, axis=0)
            all_boxes.append(array[:, :-1])
    anno_array = np.concatenate(all_boxes, axis=0)
    clusters = cluster_anchors(anno_array, 9)
    print(clusters)'''
