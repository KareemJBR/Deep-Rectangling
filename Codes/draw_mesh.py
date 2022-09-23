import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import constant
from model import rectangling_network
from utils import load, DataLoader
from entities import draw_mesh_on_warp

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

test_folder = constant.TEST_FOLDER
batch_size = constant.TEST_BATCH_SIZE
grid_w = constant.GRID_W
grid_h = constant.GRID_H


snapshot_dir = './checkpoints/pretrained_model/model.ckpt-100000'

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
        test_warp_mask_final = rectangling_network(test_input, test_mask)

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


    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = 519  # len(os.listdir(test_folder+"/input"))

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)

            mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final = \
                sess.run([test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final,
                          test_warp_image_final, test_warp_mask_final],
                         feed_dict={test_inputs_clips_tensor: input_clip})

            mesh = mesh_final[0]
            input_image = (input_clip[0, :, :, 0:3] + 1) / 2 * 255

            input_image = draw_mesh_on_warp(input_image, mesh)
            path = "../final_mesh/" + str(i + 1).zfill(5) + ".jpg"
            cv.imwrite(path, input_image)

            print('i = {} / {}'.format(i + 1, length))


    inference_func(snapshot_dir)
