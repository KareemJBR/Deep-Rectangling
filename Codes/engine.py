import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import numpy as np


# receive two images and create augmented images
def augment_images(input_img, output_img, mask):
    # flip
    input_img_flip = np.fliplr(input_img)
    output_img_flip = np.fliplr(output_img)
    mask_flip = np.fliplr(mask)

    return input_img_flip, output_img_flip, mask_flip


# extract image mesh
def extract_mesh(input_img, output_img, mask):
    # extract mesh
    input_img_mesh = input_img[:, :, 0:3]
    output_img_mesh = output_img[:, :, 0:3]
    mask_mesh = mask[:, :, 0:3]

    return input_img_mesh, output_img_mesh, mask_mesh


if __name__ == "__main__":
    data = []
    for i in range(10):
        data.append(("../DIR-D/training/input/{}".format(i), "../DIR-D/training/output/{}".format(i),
                    "../DIR-D/training/mask/{}".format(i)))

    for input_img, output_img, mask in data:
        input_img = cv2.imread(input_img)
        output_img = cv2.imread(output_img)
        mask = cv2.imread(mask)
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #
        # new_input, new_output, new_mask = augment_images(input_img, output_img, mask)

        plt.figure()
        plt.subplot(1, 4, 1)
        plt.imshow(input_img, cmap='RGB')
        plt.subplot(1, 4, 2)
        plt.imshow(output_img, cmap='RGB')
        plt.subplot(1, 4, 3)
        # plt.imshow(new_input)
        # plt.subplot(1, 4, 4)
        # plt.imshow(new_output)
        plt.show()
