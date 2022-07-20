import tensorflow as tf
import numpy as np

#######################################################
# Auxiliary matrices used to solve DLT
Aux_M1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

Aux_M2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)

Aux_M3 = np.array([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1]], dtype=np.float64)

Aux_M4 = np.array([
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)

Aux_M5 = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)

Aux_M6 = np.array([
    [-1],
    [0],
    [-1],
    [0],
    [-1],
    [0],
    [-1],
    [0]], dtype=np.float64)

Aux_M71 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

Aux_M72 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, -1, 0]], dtype=np.float64)

Aux_M8 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, -1]], dtype=np.float64)

Aux_Mb = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)


########################################################

def solve_DLT(orig_pt4, pred_pt4):
    batch_size = tf.shape(orig_pt4)[0]
    orig_pt4 = tf.expand_dims(orig_pt4, [2])
    pred_pt4 = tf.expand_dims(pred_pt4, [2])

    # Auxiliary tensors used to create Ax = b equation
    m1 = tf.constant(Aux_M1, tf.float32)
    m1_tensor = tf.expand_dims(m1, [0])
    m1_tile = tf.tile(m1_tensor, [batch_size, 1, 1])

    m2 = tf.constant(Aux_M2, tf.float32)
    m2_tensor = tf.expand_dims(m2, [0])
    m2_tile = tf.tile(m2_tensor, [batch_size, 1, 1])

    m3 = tf.constant(Aux_M3, tf.float32)
    m3_tensor = tf.expand_dims(m3, [0])
    m3_tile = tf.tile(m3_tensor, [batch_size, 1, 1])

    m4 = tf.constant(Aux_M4, tf.float32)
    m4_tensor = tf.expand_dims(m4, [0])
    m4_tile = tf.tile(m4_tensor, [batch_size, 1, 1])

    m5 = tf.constant(Aux_M5, tf.float32)
    m5_tensor = tf.expand_dims(m5, [0])
    m5_tile = tf.tile(m5_tensor, [batch_size, 1, 1])

    m6 = tf.constant(Aux_M6, tf.float32)
    m6_tensor = tf.expand_dims(m6, [0])
    m6_tile = tf.tile(m6_tensor, [batch_size, 1, 1])

    m71 = tf.constant(Aux_M71, tf.float32)
    m71_tensor = tf.expand_dims(m71, [0])
    m71_tile = tf.tile(m71_tensor, [batch_size, 1, 1])

    m72 = tf.constant(Aux_M72, tf.float32)
    m72_tensor = tf.expand_dims(m72, [0])
    m72_tile = tf.tile(m72_tensor, [batch_size, 1, 1])

    m8 = tf.constant(Aux_M8, tf.float32)
    m8_tensor = tf.expand_dims(m8, [0])
    m8_tile = tf.tile(m8_tensor, [batch_size, 1, 1])

    mb = tf.constant(Aux_Mb, tf.float32)
    mb_tensor = tf.expand_dims(mb, [0])
    mb_tile = tf.tile(mb_tensor, [batch_size, 1, 1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    a1 = tf.matmul(m1_tile, orig_pt4)  # Column 1
    a2 = tf.matmul(m2_tile, orig_pt4)  # Column 2
    a3 = m3_tile  # Column 3
    a4 = tf.matmul(m4_tile, orig_pt4)  # Column 4
    a5 = tf.matmul(m5_tile, orig_pt4)  # Column 5
    a6 = m6_tile  # Column 6
    a7 = tf.matmul(m71_tile, pred_pt4) * tf.matmul(m72_tile, orig_pt4)  # Column 7
    a8 = tf.matmul(m71_tile, pred_pt4) * tf.matmul(m8_tile, orig_pt4)  # Column 8

    # tmp = tf.reshape(a1, [-1, 8])  #batch_size * 8
    # A_mat: batch_size * 8 * 8          a1-A8相当�?*8中的每一�?
    a_mat = tf.transpose(tf.stack([tf.reshape(a1, [-1, 8]), tf.reshape(a2, [-1, 8]),
                                   tf.reshape(a3, [-1, 8]), tf.reshape(a4, [-1, 8]),
                                   tf.reshape(a5, [-1, 8]), tf.reshape(a6, [-1, 8]),
                                   tf.reshape(a7, [-1, 8]), tf.reshape(a8, [-1, 8])], axis=1),
                         perm=[0, 2, 1])  # BATCH_SIZE x 8 (A_i) x 8
    print('--Shape of A_mat:', a_mat.get_shape().as_list())
    # Form b matrix
    b_mat = tf.matmul(mb_tile, pred_pt4)
    print('--shape of b:', b_mat.get_shape().as_list())

    # Solve the Ax = b
    h_8el = tf.matrix_solve(a_mat, b_mat)  # BATCH_SIZE x 8.
    print('--shape of H_8el', h_8el)

    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = tf.ones([batch_size, 1, 1])
    h_9el = tf.concat([h_8el, h_ones], 1)
    h_flat = tf.reshape(h_9el, [-1, 9])
    # H_mat = tf.reshape(h_flat ,[-1 ,3 ,3])   # BATCH_SIZE x 3 x 3
    return h_flat
