import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import functions
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import LineString, Point
import sys
sys.path.insert(1, '/Users/megbarya/Downloads/Deep-Rectangling/Codes')

from Codes.utils import load, DataLoader
from Codes.model import RectanglingNetwork
from Codes import constant



os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

test_folder = '../Codes/DIR-D/testing'
batch_size = constant.TEST_BATCH_SIZE
grid_w = constant.GRID_W
grid_h = constant.GRID_H


def draw_mesh_on_warp(warp, f_local,A,B):
    # f_local[3,0,0] = f_local[3,0,0] - 2
    # f_local[4,0,0] = f_local[4,0,0] - 4
    # f_local[5,0,0] = f_local[5,0,0] - 6
    # f_local[6,0,0] = f_local[6,0,0] - 8
    # f_local[6,0,1] = f_local[6,0,1] + 7

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
            if (j == grid_w and i == grid_h) or (j < A[0] or j > B[1]) or (i < A[1] or i > B[0]) or (j == B[1] and i == B[0]):
                continue
            elif j == grid_w or j == B[1]:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                        constant.BLUE, thickness, line_type)
                pts_ls.append((f_local[i, j, 0], f_local[i, j, 1]))
                pts_ls.append((f_local[i + 1, j, 0], f_local[i + 1, j, 1]))
                # cv.circle(warp, (f_local[i, j, 0], f_local[i, j, 1]), radius=5, color=(0, 0, 255), thickness=3)
                # cv.circle(warp, (f_local[i + 1, j, 0], f_local[i + 1, j, 1]), radius=3, color=(0, 0, 255), thickness=5)
            elif i == grid_h or i == B[0]:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                        constant.BLUE, thickness, line_type)
                pts_ls.append((f_local[i, j, 0], f_local[i, j, 1]))
                pts_ls.append((f_local[i, j + 1, 0], f_local[i, j + 1, 1]))
                # cv.circle(warp, (f_local[i, j, 0], f_local[i, j, 1]), radius=5, color=(0, 0, 255), thickness=3)
                # cv.circle(warp, (f_local[i, j + 1, 0], f_local[i, j + 1, 1]), radius=3, color=(0, 0, 255), thickness=5)
            else:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                        constant.BLUE, thickness, line_type)
                pts_ls.append((f_local[i, j, 0], f_local[i, j, 1]))
                pts_ls.append((f_local[i + 1, j, 0], f_local[i + 1, j, 1]))
                # cv.circle(warp, (f_local[i, j, 0], f_local[i, j, 1]), radius=5, color=(0, 0, 255), thickness=3)
                # cv.circle(warp, (f_local[i + 1, j, 0], f_local[i + 1, j, 1]), radius=3, color=(0, 0, 255), thickness=5)

                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                        constant.BLUE, thickness, line_type)
                pts_ls.append((f_local[i, j, 0], f_local[i, j, 1]))
                pts_ls.append((f_local[i, j + 1, 0], f_local[i, j + 1, 1]))
                # cv.circle(warp, (f_local[i, j, 0], f_local[i, j, 1]), radius=5, color=(0, 0, 255), thickness=5)
                # cv.circle(warp, (f_local[i, j + 1, 0], f_local[i, j + 1, 1]), radius=3, color=(0, 0, 255), thickness=3)

    return warp ,pts_ls



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


    def cropByMesh(input_image):
        # create the mask
        mask = cv.inRange(input_image, constant.BLUE, constant.BLUE)

        # get bounds of white pixels
        white = np.where(mask == 255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        print(xmin, xmax, ymin, ymax)

        # crop the image at the bounds
        crop = input_image[ymin:ymax, xmin:xmax]
        return crop


    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = 12  # len(os.listdir(test_folder+"/input"))

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)

            mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final = \
                sess.run([test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final,
                          test_warp_image_final, test_warp_mask_final],
                         feed_dict={test_inputs_clips_tensor: input_clip})

            mesh = mesh_final[0]
            input_image = (input_clip[0, :, :, 0:3] + 1) / 2 * 255

            input_image, pts = draw_mesh_on_warp(input_image, mesh,[2,2],[5,5])
            # input_mask = draw_mesh_on_warp(np.ones([384, 512, 3], np.int32)*255, mesh)

            path = "./Our_Final_Mesh/" + str(i + 1).zfill(5) + ".jpg"
            cropped_path = "./Our_Final_Mesh_cropped/" + str(i + 1).zfill(5) + ".jpg"
            cropped_image = cropByMesh(input_image)
            cv.imwrite(path, input_image)
            cv.imwrite(cropped_path, cropped_image)

            # plt.figure()
            # plt.imshow(crop)
            # # plt.plot(xs, ys)
            # plt.show()

            # path = "../mesh_mask/" + str(i+1).zfill(5) + ".jpg"
            # cv.imwrite(path, input_mask)

            print('i = {} / {}'.format(i + 1, length))


    inference_func(snapshot_dir)
    plt.figure()
    img = cv.imread("./Our_Final_Mesh/00011.jpg")

    plt.imshow(img)
    plt.show()
