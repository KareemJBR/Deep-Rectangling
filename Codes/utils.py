import tensorflow as tf
import numpy as np
from collections import OrderedDict
import sys
import os
import glob
import cv2
import constant
from skimage.transform import resize

rng = np.random.RandomState(2017)
grid_h = constant.GRID_H
grid_w = constant.GRID_W


class DataLoader(object):
    def __init__(self, data_folder):
        self.dir = data_folder
        self.datas = OrderedDict()
        self.setup()

    def __call__(self, batch_size):
        data_info_list = list(self.datas.values())
        length = data_info_list[0]['length']

        def data_clip_generator():
            while True:
                data_clip = []
                frame_id = rng.randint(0, length - 1)
                curr_mesh = np.load('./DIR-D/training/mesh/' + str(frame_id).zfill(5) + ".npy")
                # inputs

                input_img = np_load_frame(data_info_list[1]['frame'][frame_id], 384, 512)
                mask_img = np_load_frame(data_info_list[2]['frame'][frame_id], 384, 512)
                gt_img = np_load_frame(data_info_list[0]['frame'][frame_id], 384, 512)

                data_clip.append(input_img)
                data_clip.append(mask_img)
                data_clip.append(gt_img)
                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

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

                cropped_input, cropped_gt, cropped_mask = get_cropped(self, frame_id, curr_mesh, [0, 0], [3, 8])

                data_clip.append(cropped_input)
                data_clip.append(cropped_mask)
                data_clip.append(cropped_gt)

                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

                # second crop window

                data_clip = []

                cropped_input, cropped_gt, cropped_mask = get_cropped(self, frame_id, curr_mesh, [0, 0], [6, 4])

                data_clip.append(cropped_input)
                data_clip.append(cropped_mask)
                data_clip.append(cropped_gt)

                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

                # third crop window

                data_clip = []

                cropped_input, cropped_gt, cropped_mask = get_cropped(self, frame_id, curr_mesh, [3, 0], [6, 8])

                data_clip.append(cropped_input)
                data_clip.append(cropped_mask)
                data_clip.append(cropped_gt)
                data_clip = np.concatenate(data_clip, axis=2)

                yield data_clip

                # fourth crop window

                data_clip = []

                cropped_input, cropped_gt, cropped_mask = get_cropped(self, frame_id, curr_mesh, [0, 4], [6, 8])

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


def draw_mesh_on_warp(warp, f_local, top_left, bottom_right):
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

    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            num = num + 1

            if (j == grid_w and i == grid_h) or (j < top_left[0] or j > bottom_right[1]) or (i < top_left[1] or
                                                                                             i > bottom_right[0]) or (
                    j == bottom_right[1] and i == bottom_right[0]):
                continue

            elif j == grid_w or j == bottom_right[1]:
                cv2.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                         constant.BLUE, thickness, line_type)

            elif i == grid_h or i == bottom_right[0]:
                cv2.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                         constant.BLUE, thickness, line_type)

            else:
                cv2.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                         constant.BLUE, thickness, line_type)

    return warp


def crop_by_mesh(input_image, mesh_image, input_mask, gt_img):
    # create the mask
    mask = cv2.inRange(mesh_image, constant.BLUE, constant.BLUE)
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
        cv2.line(img, (x1, y1), (x2, y2), constant.BLUE, 1)

    for i in range(step_cols):
        [x1_, y1_, x2_, y2_] = h_xy[i]
        cv2.line(img, (x1_, y1_), (x2_, y2_), constant.BLUE, 1)

    return img


def tensor_to_np_mat(tensor):
    rows, cols, channels = tensor.shape
    res = np.zeros(shape=(rows, cols, channels))

    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                res[i][j][k] = tensor[i][j][k]

    return res


def get_cropped(dataloader, frame_id, mesh, top_left, bottom_right):
    data_info_list = list(dataloader.datas.values())

    input_image = np_load_frame(data_info_list[1]['frame'][frame_id], 384, 512)
    mask_image = np_load_frame(data_info_list[2]['frame'][frame_id], 384, 512)
    gt_image = np_load_frame(data_info_list[0]['frame'][frame_id], 384, 512)

    source_input_img = np_load_frame(data_info_list[1]['frame'][frame_id], 384, 512)
    mesh_input_img = draw_mesh_on_warp(input_image, mesh, top_left, bottom_right)

    cropped_img, cropped_mesh, cropped_mask, cropped_gt = \
        crop_by_mesh(source_input_img, mesh_input_img, mask_image, gt_image)

    cropped_img, cropped_gt, cropped_mask = \
        tensor_to_np_mat(cropped_img), tensor_to_np_mat(cropped_gt), tensor_to_np_mat(cropped_mask)

    cropped_img = resize(cropped_img, (384, 512, 3))
    cropped_gt = resize(cropped_gt, (384, 512, 3))
    cropped_mask = resize(cropped_mask, (384, 512, 3))

    return cropped_img, cropped_gt, cropped_mask
