import tensorflow as tf
import numpy as np
from collections import OrderedDict
import sys
import os
import glob
import cv2
import constant
from model import rectangling_network


rng = np.random.RandomState(2017)


class DataLoader(object):
    def __init__(self, data_folder):
        self.dir = data_folder
        self.datas = OrderedDict()
        self.setup()

    def __call__(self, batch_size):
        data_info_list = list(self.datas.values())
        length = data_info_list[0]['length']

        def data_clip_generator():
            curr_mesh = []
            while True:
                curr_mesh = []
                data_clip = []
                frame_id = rng.randint(0, length - 1)
                # inputs

                input_img = np_load_frame(data_info_list[1]['frame'][frame_id], 384, 512)
                mask_img = np_load_frame(data_info_list[2]['frame'][frame_id], 384, 512)
                gt_img = np_load_frame(data_info_list[0]['frame'][frame_id], 384, 512)

                data_clip.append(input_img)
                data_clip.append(mask_img)
                data_clip.append(gt_img)
                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

                # TODO: add tf session

                # define dataset
                with tf.name_scope('dataset'):
                    # ----------- testing ----------- #
                    train_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3 * 3], dtype=tf.float32)

                    train_input = train_inputs_clips_tensor[..., 0:3]
                    train_mask = train_inputs_clips_tensor[..., 3:6]
                    train_gt = train_inputs_clips_tensor[..., 6:9]

                    print('train input = {}'.format(train_input))
                    print('train mask = {}'.format(train_mask))
                    print('train gt = {}'.format(train_gt))

                # define testing generator function
                with tf.variable_scope('generator', reuse=None):
                    print('training = {}'.format(tf.get_variable_scope().name))
                    train_mesh_primary, train_warp_image_primary, train_warp_mask_primary, train_mesh_final, \
                        train_warp_image_final, train_warp_mask_final = rectangling_network(train_input, train_mask)

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                with tf.Session(config=config) as sess:
                    # dataset
                    input_loader = DataLoader(constant.TRAIN_FOLDER)

                    # initialize weights
                    sess.run(tf.global_variables_initializer())
                    print('Init global successfully!')

                    def inference_func():
                        for i in range(0, length):
                            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)

                            mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, \
                                warp_mask_final = sess.run([train_mesh_primary, train_warp_image_primary,
                                                            train_warp_mask_primary, train_mesh_final,
                                                            train_warp_image_final, train_warp_mask_final],
                                                           feed_dict={train_inputs_clips_tensor: input_clip})

                            curr_mesh.append(mesh_final[0])

                    inference_func()

                # creating augmentations

                data_clip = []

                # flipped augmentation
                flipped_input = np.fliplr(input_img)
                flipped_mask = np.fliplr(mask_img)
                flipped_gt = np.fliplr(gt_img)

                data_clip.append(flipped_input)
                data_clip.append(flipped_mask)
                data_clip.append(flipped_gt)
                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

                # cropped augmentations:
                # first crop window

                data_clip = []

                cropped_input, cropped_gt, cropped_mask = get_cropped(self, frame_id, curr_mesh[0])

                data_clip.append(cropped_input)
                data_clip.append(cropped_mask)
                data_clip.append(cropped_gt)

                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

                # second crop window

                data_clip = []

                cropped_input, cropped_gt, cropped_mask = get_cropped(self, frame_id, curr_mesh[0])

                data_clip.append(cropped_input)
                data_clip.append(cropped_mask)
                data_clip.append(cropped_gt)

                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

                # third crop window

                data_clip = []

                cropped_input, cropped_gt, cropped_mask = get_cropped(self, frame_id, curr_mesh[0])

                data_clip.append(cropped_input)
                data_clip.append(cropped_mask)
                data_clip.append(cropped_gt)
                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

                # fourth crop window

                data_clip = []

                cropped_input, cropped_gt, cropped_mask = get_cropped(self, frame_id, curr_mesh[0])

                data_clip.append(cropped_input)
                data_clip.append(cropped_mask)
                data_clip.append(cropped_gt)
                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

        dataset = tf.data.Dataset.from_generator(generator=data_clip_generator, output_types=tf.float32,
                                                 output_shapes=[384, 512, 9])

        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=128)
        dataset = dataset.shuffle(buffer_size=128).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, data_name):
        assert data_name in self.datas.keys(), 'data = {} is not in {}!'.format(data_name, self.datas.keys())
        return self.datas[data_name]

    def setup(self):
        datas = glob.glob(os.path.join(self.dir, '*'))
        for data in sorted(datas):

            if sys.platform[:3] == 'win':
                data_name = data.split('\\')[-1]
            else:
                data_name = data.split('/')[-1]

            if data_name == 'gt' or data_name == 'input' or data_name == 'mask':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['frame'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['frame'].sort()
                self.datas[data_name]['length'] = len(self.datas[data_name]['frame'])

        print(self.datas.keys())

    def get_data_clips(self, index):
        batch = []
        data_info_list = list(self.datas.values())

        batch.append(np_load_frame(data_info_list[1]['frame'][index], 384, 512))
        batch.append(np_load_frame(data_info_list[2]['frame'][index], 384, 512))
        batch.append(np_load_frame(data_info_list[0]['frame'][index], 384, 512))

        return np.concatenate(batch, axis=2)


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


def crop_by_mesh(input_image, mesh_image, input_mask, gt_img):
    # create the mask
    mask = cv2.inRange(mesh_image, constant.BLUE, constant.BLUE)
    # get bounds of white pixels
    white = np.where(mask == 255)

    # TODO: causes errors
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
        cv2.line(img, (x1, y1), (x2, y2), constant.BLUE, 1)

    for i in range(step_cols):
        [x1_, y1_, x2_, y2_] = h_xy[i]
        cv2.line(img, (x1_, y1_), (x2_, y2_), constant.BLUE, 1)

    return img


def get_cropped(dataloader, frame_id, mesh):
    data_info_list = list(dataloader.datas.values())

    input_image = np_load_frame(data_info_list[1]['frame'][frame_id], 384, 512)
    mask_image = np_load_frame(data_info_list[2]['frame'][frame_id], 384, 512)
    gt_image = np_load_frame(data_info_list[0]['frame'][frame_id], 384, 512)

    source_input_img = np_load_frame(data_info_list[1]['frame'][frame_id], 384, 512)
    mesh_input_img = draw_mesh_on_warp(input_image, mesh)

    cropped_img, cropped_mesh, cropped_mask, cropped_gt = \
        crop_by_mesh(source_input_img, mesh_input_img, mask_image, gt_image)

    return cropped_img, cropped_gt, cropped_mask
