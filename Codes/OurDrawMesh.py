import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import sys

from utils import load, DataLoader
from model import RectanglingNetwork
import constant

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

test_folder = '../Codes/DIR-D/testing'
batch_size = constant.TEST_BATCH_SIZE
grid_w = constant.GRID_W
grid_h = constant.GRID_H


def draw_mesh_on_warp(warp, f_local, A, B):
    min_w = np.minimum(np.min(f_local[:, :, 0]), 0).astype(np.int32)
    max_w = np.maximum(np.max(f_local[:, :, 0]), 512).astype(np.int32)
    min_h = np.minimum(np.min(f_local[:, :, 1]), 0).astype(np.int32)
    max_h = np.maximum(np.max(f_local[:, :, 1]), 384).astype(np.int32)
    cw = max_w - min_w
    ch = max_h - min_h

    pic = np.ones([ch + 10, cw + 10, 3], np.int32) * 255
    pic[0 - min_h + 5:0 - min_h + 384 + 5, 0 - min_w + 5:0 - min_w + 512 + 5, :] = warp

    warp = pic
    f_local[:, :, 0] = f_local[:, :, 0] - min_w + 5
    f_local[:, :, 1] = f_local[:, :, 1] - min_h + 5

    thickness = 2
    line_type = 8
    num = 1

    pts_ls = []

    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            num = num + 1
            if (j == grid_w and i == grid_h) or (j < A[0] or j > B[1]) or (i < A[1] or i > B[0]) or (
                    j == B[1] and i == B[0]):
                continue
            elif j == grid_w or j == B[1]:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                        constant.BLUE, thickness, line_type)
                pts_ls.append((f_local[i, j, 0], f_local[i, j, 1]))
                pts_ls.append((f_local[i + 1, j, 0], f_local[i + 1, j, 1]))
            elif i == grid_h or i == B[0]:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                        constant.BLUE, thickness, line_type)
                pts_ls.append((f_local[i, j, 0], f_local[i, j, 1]))
                pts_ls.append((f_local[i, j + 1, 0], f_local[i, j + 1, 1]))
            else:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                        constant.BLUE, thickness, line_type)
                pts_ls.append((f_local[i, j, 0], f_local[i, j, 1]))
                pts_ls.append((f_local[i + 1, j, 0], f_local[i + 1, j, 1]))
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                        constant.BLUE, thickness, line_type)
                pts_ls.append((f_local[i, j, 0], f_local[i, j, 1]))
                pts_ls.append((f_local[i, j + 1, 0], f_local[i, j + 1, 1]))

    return warp, pts_ls


snapshot_dir = '../Codes/checkpoints/pretrained_model/model.ckpt-100000'

# define dataset
with tf.name_scope('dataset'):
    # ----------- testing ----------- #
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3 * 3], dtype=tf.float32)

    test_input = test_inputs_clips_tensor[..., 0:3]
    test_mask = test_inputs_clips_tensor[..., 3:6]
    test_gt = test_inputs_clips_tensor[..., 6:9]

    print('test input = {}'.format(test_input))
    print('test mask = {}'.format(test_mask))
    print('test gt = {}'.format(test_gt))

# define testing generator function
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final, test_warp_image_final, \
        test_warp_mask_final = RectanglingNetwork(test_input, test_mask)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # dataset
    input_loader = DataLoader(test_folder)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)


    def fixYaxle(pts):
        xs, ys = zip(*pts)
        xs, ys = list(xs), list(ys)
        for i1 in range(len(ys)):
            # xs[i] = 512-xs[i]
            ys[i1] = 384 - ys[i1]

        return ys


    def cropByMesh(input_image, mesh_image, input_mask, gt_img):
        # create the mask
        mask = cv.inRange(mesh_image, constant.BLUE, constant.BLUE)
        cv.imwrite("mask.jpg", mask)
        # get bounds of white pixels
        white = np.where(mask == 255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        print(xmin, xmax, ymin, ymax)

        # crop the image at the bounds
        cropped_mesh = mesh_image[ymin:ymax, xmin:xmax]
        cropped_input = input_image[ymin:ymax, xmin:xmax]
        cropped_mask = input_mask[ymin:ymax, xmin:xmax]
        cropped_gt = gt_img[ymin:ymax, xmin:xmax]
        return cropped_input, cropped_mesh, cropped_mask, cropped_gt


    def draw_grid(img):
        # h, w, _ = img.shape
        # dy, dx = h / grid_h , w / grid_w
        #
        # for x in np.linspace(start=dx, stop=w - dx, num=grid_h - 1):
        #     cv.line(img, (x, 0), (x, h), color=constant.BLUE, thickness=2)
        #
        # for y in np.linspace(start=dy, stop=h - dy, num=grid_w - 1):
        #     cv.line(img, (0, y), (w, y), color=constant.BLUE, thickness=2)

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


    def add_borders_and_resize(image, type):
        if type is 'gt':
            return image

        if type is 'mask':
            borderColor = 0
        else:
            borderColor = (255, 255, 255)

        withborder = cv.copyMakeBorder(image, top=10, bottom=10, left=10, right=10, borderType=cv.BORDER_CONSTANT,
                                       value=borderColor
                                       )

        return withborder


    def get_cropped(ckpt, index, A, B):
        print("====================tom tst st tst sts  ta  tach tach====")
        print(ckpt)
        load(loader, sess, ckpt)
        print("====================tom tst st tst sts  ta  tach tach====")

        input_clip = np.expand_dims(input_loader.get_data_clips(index), axis=0)

        mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final = sess.run(
            [test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final,
             test_warp_image_final, test_warp_mask_final], feed_dict={test_inputs_clips_tensor: input_clip})

        print(mesh_primary.shape)
        print(mesh_primary)
        mesh = mesh_final[0]
        input_image = (input_clip[0, :, :, 0:3] + 1) / 2 * 255
        gt_img = (input_clip[0, :, :, 6:9] + 1) / 2 * 255
        mask_img = (input_clip[0, :, :, 3:6] + 1) / 2 * 255

        source_input_img = (input_clip[0, :, :, 0:3] + 1) / 2 * 255
        mesh_input_img, pts = draw_mesh_on_warp(input_image, mesh, A, B)

        cropped_img, cropped_mesh, cropped_mask, cropped_gt = cropByMesh(source_input_img, mesh_input_img,
                                                                         mask_img,
                                                                         gt_img)

        return cropped_img, cropped_gt, cropped_mask
