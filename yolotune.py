from tracking.yolo import YOLO
import os
import multiprocessing
import cv2 as cv
import numpy as np
from math import ceil
from keras.layers import Conv2D, Lambda
from keras import Model
from tracking.yolo3.model import yolo_head
from keras import backend as K

train_data_dir = r'/home/patrick/PatrickWorkspace/Datasets/VisDrone/VisDrone2019-DET-train'
val_data_dir = r'/home/patrick/PatrickWorkspace/Datasets/VisDrone/VisDrone2019-DET-val'
img_dir = os.path.join(train_data_dir, 'images')
anno_dir = os.path.join(train_data_dir, 'annotations')
model_base = '/home/patrick/PatrickWorkspace/DroneTracking/tracking/model_data'


def scale_anchors(scale):
    assert scale == 0 or scale == 1 or scale == 2
    # TODO regression on VisDrone
    anchors = [[(10, 13), (16, 30), (33, 23)],
               [(30, 61), (62, 45), (59, 119)],
               [(116, 90), (156, 198), (373, 326)]]
    return anchors[scale]


def parse_anno_line(line, size_limit, offset=(0, 0)):
    # w,h
    str_nums = line.split(',')
    nums = []
    for str_num in str_nums:
        nums.append(int(str_num))
    if nums[5] != 0 and nums[4] == 1:
        return None
    corrected_x, corrected_y = nums[0] - offset[0], nums[1] - offset[1]
    if corrected_x < 0 or corrected_x + nums[2] >= size_limit[0]:
        return None
    if corrected_y < 0 or corrected_y + nums[3] >= size_limit[1]:
        return None
    result = ([corrected_x, corrected_y] + nums[2:4]).append(nums[5])  # bounding box and class
    return result


def data_generator(ids, img_array, anno_array, lock):
    for id in ids:
        img = cv.imread(os.path.join(img_dir, id + '.jpg'))
        # random crop
        img_height, img_width = img.shape[:-1]
        height_diff = img_height % 32
        width_diff = img_width % 32
        if img_height / 32 % 2 == 0:
            height_diff += 32
        if img_width / 32 % 2 == 0:
            width_diff += 32
        h_offset = np.random.randint(0, height_diff + 1)
        w_offset = np.random.randint(0, width_diff + 1)
        img = img[h_offset:, w_offset:, :].copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.float32(img) / 255
        # annotation
        anno = []
        with open(os.path.join(anno_dir, id + '.txt')) as anno_file:
            for line in anno_file.readlines():
                nums = parse_anno_line(line, (img_width - w_offset, img_height - h_offset), (w_offset, h_offset))
                if nums is not None:
                    anno.append(nums)
        anno = np.float32(np.array(anno))
        lock.acquire()
        img_array.append(img)
        anno_array.append(anno)
        lock.release()


def training_data(multi_proc=4):
    # TODO change to generator
    img_array = multiprocessing.Manager().list()
    anno_array = multiprocessing.Manager().list()
    lock = multiprocessing.Manager().Lock()
    fileids = [filename[:-4] for filename in os.listdir(img_dir)]
    length = ceil(len(fileids) / float(multi_proc))
    start = 0
    proc_pool = multiprocessing.Pool(multi_proc)
    while start < len(fileids):
        end = min(len(fileids), start + length)
        proc_pool.apply_async(data_generator, args=(fileids[start:end], img_array, anno_array, lock))
        start += length
    proc_pool.close()
    proc_pool.join()
    # img_ndarray = np.concatenate(img_array, axis=0)
    # anno_ndarray = np.concatenate(anno_array, axis=0)
    return img_array, anno_array


def sscale_loss(modelout, annoarray):
    anchors = scale_anchors(0)
    return multi_task_loss(modelout, anchors, annoarray)


def mscale_loss(modelout, annoarray):
    anchors = scale_anchors(1)
    return multi_task_loss(modelout, anchors, annoarray)


def lscale_loss(modelout, annoarray):
    anchors = scale_anchors(2)
    return multi_task_loss(modelout, anchors, annoarray)


def multi_task_loss(conv_output, anchors, annoarray, ignore_thresh=.5):
    input_shape = K.cast(K.shape(conv_output)[1:3] * 32, K.dtype(annoarray))
    grid_shapes = K.cast(K.shape(conv_output)[1:3], K.dtype(annoarray))
    loss = 0
    m = K.shape(conv_output)[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(conv_output))
    object_mask = annoarray[..., 4:5]
    true_class_probs = annoarray[..., ]

    grid, raw_pred, pred_xy, pred_wh = yolo_head(conv_output[l],
                                                 anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    raw_true_xy = annoarray[l][..., :2] * grid_shapes[l][::-1] - grid
    raw_true_wh = K.log(annoarray[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
    box_loss_scale = 2 - annoarray[l][..., 2:3] * annoarray[l][..., 3:4]

    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(K.dtype(annoarray[0]), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')

    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(annoarray[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
        return b + 1, ignore_mask

    _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)

    # K.binary_crossentropy is helpful to avoid exp overflow.
    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                   from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                      (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf
    loss += xy_loss + wh_loss + confidence_loss + class_loss
    return loss

def tune_yolo():
    # train_array, train_label = training_data()
    model = YOLO(model_base).yolo_model
    sscale_relu = model.get_layer('leaky_re_lu_58').output
    mscale_relu = model.get_layer('leaky_re_lu_65').output
    lscale_relu = model.get_layer('leaky_re_lu_72').output
    input_layer = model.input
    # output for VisDrone
    sscale_out = Conv2D(filters=48, kernel_size=1, padding='same', name='conv2d_sout')(sscale_relu)
    mscale_out = Conv2D(filters=48, kernel_size=1, padding='same', name='conv2d_mout')(mscale_relu)
    lscale_out = Conv2D(filters=48, kernel_size=1, padding='same', name='conv2d_lout')(lscale_relu)
    new_model = Model(inputs=input_layer, outputs=[sscale_out, mscale_out, lscale_out])
    print(new_model.summary())


if __name__ == '__main__':
    tune_yolo()
