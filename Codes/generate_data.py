import copy
# import seam_carving
import glob
import os
import cv2

if __name__ == "__main__":
    gt_dir = './DIR-D/training/gt'
    input_dir = './DIR-D/training/input'
    mask_dir = './DIR-D/training/mask'

    gt_files = glob.glob(os.path.join(gt_dir, '*.jpg'))
    input_files = glob.glob(os.path.join(input_dir, '*.jpg'))
    mask_files = glob.glob(os.path.join(mask_dir, '*.jpg'))

    img_num = 1
    file_dir = './DIR-D/new_training/'

    for i in range(len(gt_files)):
        gt_file = gt_files[i]
        input_file = input_files[i]
        mask_file = mask_files[i]

        gt_img = cv2.imread(gt_file)
        input_img = cv2.imread(input_file)
        mask_img = cv2.imread(mask_file)

        curr_gt = copy.deepcopy(gt_img)
        curr_input = copy.deepcopy(input_img)
        curr_mask = copy.deepcopy(mask_img)

        for _ in range(3):
            curr_gt = curr_gt.transpose(1, 0, 2)
            curr_input = curr_input.transpose(1, 0, 2)
            curr_mask = curr_mask.transpose(1, 0, 2)

            str_num = str(img_num)
            while len(str_num) != 5:
                str_num = '0' + str_num

            cv2.imwrite(file_dir + 'gt/' + str_num + '.jpg', curr_gt)
            cv2.imwrite(file_dir + 'input/' + str_num + '.jpg', curr_input)
            cv2.imwrite(file_dir + 'mask/' + str_num + '.jpg', curr_mask)

            img_num += 1
