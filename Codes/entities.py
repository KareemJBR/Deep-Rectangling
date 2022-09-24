import numpy as np
import cv2
import constant
import os


def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)

    if resize_height is not None:
        image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    else:
        image_resized = image_decoded

    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


def load(saver, sess, ckpt_path):
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def draw_mesh_on_warp(warp, f_local, grid_h=constant.GRID_H, grid_w=constant.GRID_W):
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

    point_color = (0, 255, 0)  # BGR
    thickness = 2
    line_type = 8
    num = 1

    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                         point_color, thickness, line_type)
            elif i == grid_h:
                cv2.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                         point_color, thickness, line_type)
            else:
                cv2.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                         point_color, thickness, line_type)
                cv2.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                         point_color, thickness, line_type)

    return warp
