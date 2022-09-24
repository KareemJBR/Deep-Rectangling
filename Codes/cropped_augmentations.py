import tensorflow as tf
import numpy as np
import cv2 as cv
import constant
from entities import draw_mesh_on_warp
from model import rectangling_network


def crop_by_mesh(input_image, mesh_image, input_mask, gt_img):
    # create the mask
    mask = cv.inRange(mesh_image, constant.BLUE, constant.BLUE)
    # get bounds of white pixels
    white = np.where(mask == 255)
    x_min, y_min, x_max, y_max = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])

    # crop the image at the bounds
    cropped_mesh = mesh_image[y_min:y_max, x_min:x_max]
    cropped_input = input_image[y_min:y_max, x_min:x_max]
    cropped_mask = input_mask[y_min:y_max, x_min:x_max]
    cropped_gt = gt_img[y_min:y_max, x_min:x_max]
    return cropped_input, cropped_mesh, cropped_mask, cropped_gt


def draw_grid(img):
    rows = 384
    cols = 512

    step_rows = 6
    x = np.linspace(start=0, stop=rows, num=step_rows + 1)
    step_cols = 8
    y = np.linspace(start=0, stop=cols, num=step_cols + 1)

    v_xy = []
    h_xy = []
    for i in range(step_rows):
        v_xy.append([int(x[i]), 0, int(x[i]), rows])

    for i in range(step_cols):
        h_xy.append([0, int(y[i]), cols, int(y[i])])

    for i in range(step_rows):
        [x1, y1, x2, y2] = v_xy[i]
        cv.line(img, (x1, y1), (x2, y2), constant.BLUE, 1)

    for i in range(step_cols):
        [x1_, y1_, x2_, y2_] = h_xy[i]
        cv.line(img, (x1_, y1_), (x2_, y2_), constant.BLUE, 1)

    return img


def get_image_mesh(img, msk):
    tmp_img = tf.Tensor()

    with tf.variable_scope('generator', reuse=None):
        print('testing = {}'.format(tf.get_variable_scope().name))
        test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final, test_warp_image_final, \
            test_warp_mask_final = rectangling_network(img, msk)

    return test_mesh_final


def get_cropped(index, top_left, bottom_right):

    input_image = cv.imread("./DIR-D/training/input/0000" + str(index) + ".jpg")
    gt_image = cv.imread("./DIR-D/training/gt/0000" + str(index) + ".jpg")
    mask_image = cv.imread("./DIR-D/training/mask/0000" + str(index) + ".jpg")
    mesh = get_image_mesh(input_image, mask_image)

    source_input_img = cv.imread("./DIR-D/training/input/0000" + str(index) + ".jpg")
    mesh_input_img, pts = draw_mesh_on_warp(input_image, mesh, top_left, bottom_right)

    cropped_img, cropped_mesh, cropped_mask, cropped_gt = crop_by_mesh(source_input_img, mesh_input_img,
                                                                       mask_image,
                                                                       gt_image)

    cropped_img = cv.resize(cropped_img, dsize=(512, 384, 3), interpolation=cv.INTER_CUBIC)
    cropped_gt = cv.resize(cropped_gt, dsize=(512, 384, 3), interpolation=cv.INTER_CUBIC)
    cropped_mask = cv.resize(cropped_mask, dsize=(512, 384), interpolation=cv.INTER_CUBIC)

    return cropped_img, cropped_gt, cropped_mask
