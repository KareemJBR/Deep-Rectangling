import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import conv2d
import tf_spatial_transform_local
import tf_spatial_transform_local_feature
import constant

grid_w = constant.GRID_W
grid_h = constant.GRID_H


def shift2mesh(mesh_shift, width, height):
    batch_size = tf.shape(mesh_shift)[0]
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = tf.constant([ww, hh], shape=[2], dtype=tf.float32)
            ori_pt.append(tf.expand_dims(p, 0))
    ori_pt = tf.concat(ori_pt, axis=0)
    ori_pt = tf.reshape(ori_pt, [grid_h + 1, grid_w + 1, 2])
    ori_pt = tf.tile(tf.expand_dims(ori_pt, 0), [batch_size, 1, 1, 1])

    return ori_pt + mesh_shift


def rectangling_network(train_input, train_mask, width=512., height=384.):
    mesh_shift_primary, mesh_shift_final = build_model(train_input, train_mask)

    mesh_primary = shift2mesh(mesh_shift_primary, width, height)
    mesh_final = shift2mesh(mesh_shift_final + mesh_shift_primary, width, height)

    warp_image_primary, warp_mask_primary = tf_spatial_transform_local.transformer(train_input, train_mask,
                                                                                   mesh_primary)
    warp_image_final, warp_mask_final = tf_spatial_transform_local.transformer(train_input, train_mask, mesh_final)

    return mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final


# feature extraction module
def feature_extractor(image_tf):
    feature = []
    # 512*384
    with tf.variable_scope('conv_block1'):
        conv1 = conv2d(inputs=image_tf, num_outputs=64, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
        conv1 = conv2d(inputs=conv1, num_outputs=64, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
        maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding='SAME')
    # 256*192
    with tf.variable_scope('conv_block2'):
        conv2 = conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
        conv2 = conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
        maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding='SAME')
    # 128*96
    with tf.variable_scope('conv_block3'):
        conv3 = conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
        conv3 = conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
        maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding='SAME')
    # 64*48
    with tf.variable_scope('conv_block4'):
        conv4 = conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
        conv4 = conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
        feature.append(conv4)

    return feature


# mesh motion regression module
def regression_net(correlation):
    conv1 = conv2d(inputs=correlation, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = conv2d(inputs=conv1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)

    maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding='SAME')  # 16
    conv2 = conv2d(inputs=maxpool1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv2 = conv2d(inputs=conv2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)

    maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding='SAME')  # 8
    conv3 = conv2d(inputs=maxpool2, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv3 = conv2d(inputs=conv3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)

    maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding='SAME')  # 4
    conv4 = conv2d(inputs=maxpool3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv4 = conv2d(inputs=conv4, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)

    fc1 = conv2d(inputs=conv4, num_outputs=2048, kernel_size=[3, 4], activation_fn=tf.nn.relu, padding="VALID")
    fc2 = conv2d(inputs=fc1, num_outputs=1024, kernel_size=1, activation_fn=tf.nn.relu)
    fc3 = conv2d(inputs=fc2, num_outputs=(grid_w + 1) * (grid_h + 1) * 2, kernel_size=1, activation_fn=None)
    # net3_f = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    return tf.reshape(fc3, (-1, grid_h + 1, grid_w + 1, 2))


def build_model(train_input, train_mask):
    with tf.variable_scope('model'):
        with tf.variable_scope('feature_extract', reuse=None):
            features = feature_extractor(tf.concat([train_input, train_mask], axis=3))

        feature = tf.image.resize_images(features[-1], [24, 32], method=0)
        with tf.variable_scope('regression_coarse', reuse=None):
            mesh_shift_primary = regression_net(feature)

        with tf.variable_scope('regression_fine', reuse=None):
            mesh_primary = shift2mesh(mesh_shift_primary / 16, 32., 24.)
            feature_warp = tf_spatial_transform_local_feature.transformer(feature, mesh_primary)
            mesh_shift_final = regression_net(feature_warp)

        return mesh_shift_primary, mesh_shift_final
