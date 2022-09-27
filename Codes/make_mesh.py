"""
This file creates the mesh files for the training files and add them to the training dataset under the directory
training/mesh/
"""

import tensorflow as tf
import os
import numpy as np
import constant
from model import rectangling_network
from utils import DataLoader
from utils import load

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

train_folder = constant.TRAIN_FOLDER
batch_size = constant.TRAIN_BATCH_SIZE
grid_w = constant.GRID_W
grid_h = constant.GRID_H


snapshot_dir = './checkpoints/pretrained_model/model.ckpt-100000'

# define dataset
with tf.name_scope('dataset'):
    # ----------- training ----------- #
    train_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3 * 3], dtype=tf.float32)

    train_input = train_inputs_clips_tensor[..., 0:3]
    train_mask = train_inputs_clips_tensor[..., 3:6]
    train_gt = train_inputs_clips_tensor[..., 6:9]

    print('train input = {}'.format(train_input))
    print('train mask = {}'.format(train_mask))
    print('train gt = {}'.format(train_gt))

# define training generator function
with tf.variable_scope('generator', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    train_mesh_primary, train_warp_image_primary, train_warp_mask_primary, train_mesh_final, train_warp_image_final, \
        train_warp_mask_final = rectangling_network(train_input, train_mask)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # dataset
    input_loader = DataLoader(train_folder)

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
        length = 5839

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)

            mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final = \
                sess.run([train_mesh_primary, train_warp_image_primary, train_warp_mask_primary, train_mesh_final,
                          train_warp_image_final, train_warp_mask_final],
                         feed_dict={train_inputs_clips_tensor: input_clip})

            mesh = mesh_final[0]
            path = "./DIR-D/training/mesh/" + str(i + 1).zfill(5) + ".npy"

            np.save(path, mesh)

            print('i = {} / {}'.format(i + 1, length))


    inference_func(snapshot_dir)
