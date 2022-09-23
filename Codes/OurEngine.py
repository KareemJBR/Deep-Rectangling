import cv2 as cv2
import numpy as np
from random import randint
import functions
import matplotlib.pyplot as plt


def augment_images(_input_img, gt_img, _mask):
    # flip
    input_img_flip = np.fliplr(_input_img)
    gt_img_flip = np.fliplr(gt_img)
    mask_flip = np.fliplr(_mask)

    return input_img_flip, gt_img_flip, mask_flip


def extract_mesh(_input_img, gt_img, _mask):
    # extract mesh
    input_img_mesh = _input_img[:, :, 0:3]
    gt_img_mesh = gt_img[:, :, 0:3]
    mask_mesh = _mask[:, :, 0:3]

    return input_img_mesh, gt_img_mesh, mask_mesh


def plot_results(_input, _gt, _mask, new_input, new_gt, new_mask, aug_func):
    fig = plt.figure()
    plt.title(aug_func)
    ax11 = fig.add_subplot(231)
    ax12 = fig.add_subplot(232)
    ax13 = fig.add_subplot(233)
    ax21 = fig.add_subplot(234)
    ax22 = fig.add_subplot(235)
    ax23 = fig.add_subplot(236)

    ax11.title.set_text('input img')
    ax12.title.set_text('gt img')
    ax13.title.set_text('mask img')
    ax21.title.set_text('new_input img')
    ax22.title.set_text('new_gt img')
    ax23.title.set_text('new_mask img')

    ax11.imshow(_input)
    ax12.imshow(_gt)
    ax13.imshow(_mask)
    ax21.imshow(new_input)
    ax22.imshow(new_gt)
    ax23.imshow(new_mask)

    plt.show()


def generate_augmented_img(input_img, gt_image, mask):
    augmentations1 = ['blur_image', 'cropped_image', 'flip_image', 'change_brightness']
    randfunc = randint(0, 3)
    print(augmentations1[randfunc])
    if augmentations1[randfunc] is 'blur_image': #TODO keep in touch
        return functions.blur_image(input_img), functions.blur_image(gt_image), functions.blur_image(
            mask), 'blurred image'
    elif augmentations1[randfunc] is 'cropped_image':  # TODO genrate modern frame for cropping
        return functions.cropped_image(input_img, 20, 500, 0, 200), functions.cropped_image(gt_image, 20, 500, 0,
                                                                                            200), functions.cropped_image(mask, 20, 500, 0, 200), 'cropped image'
    elif augmentations1[randfunc] is 'flip_image':
        return functions.flip_image(input_img), functions.flip_image(gt_image), functions.flip_image(
            mask), 'flip image'
    elif augmentations1[randfunc] is 'change_brightness': #TODO keep in touch
        return functions.change_brightness(input_img, None), functions.change_brightness(
            gt_image,None), functions.change_brightness(
            mask,None), 'change brightness'
    else:
        return "something"


if __name__ == "__main__":

    data = []
    for i in range(1, 10):
        data.append(
            ("../Codes/DIR-D/training/input/0000{}.jpg".format(i), "../Codes/DIR-D/training/gt/0000{}.jpg".format(i),
             "../Codes/DIR-D/training/mask/0000{}.jpg".format(i)))
    #
    # data.append(("training/input/00010.jpg", "training/gt/00010.jpg", "training/mask/00010.jpg"))

    for input_, gt_, mask_ in data:
        input_img = cv2.imread(input_)
        gt_image = cv2.imread(gt_)
        mask = cv2.imread(mask_)

        # fig = plt.figure()
        # plt.title('flip_image')
        # ax11 = fig.add_subplot(231)
        # ax12 = fig.add_subplot(232)
        # ax13 = fig.add_subplot(233)
        #
        # ax11.title.set_text('input img')
        # ax12.title.set_text('gt img')
        # ax13.title.set_text('mask img')
        #
        # ax11.imshow(functions.change_brightness(input_img))
        # ax12.imshow(input_img)
        # ax13.imshow(functions.change_brightness(mask))
        # plt.show()

        new_input, new_gt, new_mask, aug_func = generate_augmented_img(input_img, gt_image, mask)
        plot_results(input_img, gt_image, mask, new_input, new_gt, new_mask, aug_func)
