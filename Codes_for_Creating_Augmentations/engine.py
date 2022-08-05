import cv2.cv2 as cv2
import numpy as np
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


def plot_results(first, second, third, fourth):
    plt.figure()
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(first)
    axes[0, 1].imshow(second)
    axes[1, 0].imshow(third)
    axes[1, 1].imshow(fourth)
    plt.show()


if __name__ == "__main__":

    data = []
    for i in range(1, 10):
        data.append(("training/input/0000{}.jpg".format(i), "training/gt/0000{}.jpg".format(i),
                    "training/mask/0000{}.jpg".format(i)))

    data.append(("training/input/00010.jpg", "training/gt/00010.jpg", "training/mask/00010.jpg"))

    for input_, gt_, mask_ in data:
        input_img = cv2.imread(input_)
        gt_image = cv2.imread(gt_)
        mask = cv2.imread(mask_)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        new_input, new_gt, new_mask = augment_images(input_img, gt_image, mask)
        plot_results(input_img, gt_image, new_input, new_gt)
