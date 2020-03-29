"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
import cv2 as cv


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
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    return boxed_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(img, annoarray, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    ih, iw = img.shape[:-1]
    h, w = input_shape
    # resize image
    scale_factor = min(h / ih, w / iw)
    image = cv.resize(img, dsize=(int(iw * scale_factor), int(ih * scale_factor)))
    h_offset = (h - image.shape[0]) // 2
    w_offset = (w - image.shaoe[1]) // 2
    height_limit = h_offset + image.shape[0]
    width_limit = w_offset + image.shape[1]
    image = cv.copyMakeBorder(image, top=h_offset, bottom=input_shape[0] - height_limit,
                              left=w_offset, right=input_shape[1] - width_limit, borderType=cv.BORDER_CONSTANT,
                              value=(128, 128))
    '''
    scaled_width = image.shape[1]
    width_limit = min(scaled_width, w)
    w_diff = scaled_width - w
    w_offset = 0
    if w_diff > 0:
        w_offset = np.random.randint(0, w_diff + 1)
        image = image[:, w_offset:w_offset + w, :].copy()
    elif w_diff < 0:
        image = cv.copyMakeBorder(image, 0, 0, 0, -w_diff, borderType=cv.BORDER_CONSTANT, value=(128, 128, 128))
    
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    # image = image.resize((nw, nh), Image.BICUBIC)
    image = cv.resize(img, dsize=(nw,nh))
    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = np.zeros((h, w, 3)) + 128
    # new_image = Image.new('RGB', (w, h), (128, 128, 128))
    image = cv.copyMakeBorder(image, top=dy, bottom=0, left=dx, right=0, borderType=cv.BORDER_CONSTANT,
                              value=(128, 128, 128))
    # new_image.paste(image, (dx, dy))
    # image = new_image
    '''
    # flip image or not
    flip = rand() < .5
    if flip:
        # image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = cv.flip(image, 1)
    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    # x = rgb_to_hsv(np.array(image) / 255.)
    x = cv.cvtColor(image / 255., cv.COLOR_RGB2HSV)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    # image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    image_data = cv.cvtColor(x, cv.COLOR_HSV2RGB)
    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(annoarray) > 0:
        np.random.shuffle(annoarray)
        # correction for resize
        annoarray[:, [0, 2]] = annoarray[:, [0, 2]] * scale_factor + w_offset
        annoarray[:, [1, 3]] = annoarray[:, [1, 3]] * scale_factor + h_offset
        annoarray = annoarray[np.logical_and(annoarray[:, 2] > 0, annoarray[:, 3] > 0)]  # remove x_max<=0 or y_max<=0
        annoarray[:, 0:2][annoarray[:, 0:2] < 0] = 0
        annoarray[:, 2][annoarray[:, 2] > width_limit - 1] = width_limit - 1
        annoarray[:, 3][annoarray[:, 3] > height_limit - 1] = height_limit - 1
        if flip:
            annoarray[:, [0, 2]] = w - annoarray[:, [2, 0]]
        annoarray_w = annoarray[:, 2] - annoarray[:, 0]
        annoarray_h = annoarray[:, 3] - annoarray[:, 1]
        annoarray = annoarray[np.logical_and(annoarray_w > 1, annoarray_h > 1)]  # discard invalid annoarray
        if len(annoarray) > max_boxes:
            annoarray = annoarray[:max_boxes]
        box_data[:len(annoarray)] = annoarray

    return image_data, box_data
