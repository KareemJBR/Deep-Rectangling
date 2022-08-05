import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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
    print(matplotlib.__version__)
    exit(0)
    data = []
    for i in range(10):
        data.append(("../DIR-D/training/input/0000{}.jpg".format(i), "../DIR-D/training/output/0000{}.jpg".format(i),
                    "../DIR-D/training/mask/0000{}.jpg".format(i)))

    for input_img, output_img, mask_img in data:
        input_img = cv2.imread(input_img)
        output_img = cv2.imread(output_img)
        mask = cv2.imread(mask_img)

        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #
        # new_input, new_output, new_mask = augment_images(input_img, output_img, mask)

        plt.figure()
        plt.subplot(1, 4, 1)
        plt.imshow(input_img, cmap='Accent')
        plt.subplot(1, 4, 2)
        plt.imshow(output_img, cmap='Accent')
        plt.subplot(1, 4, 3)
        # plt.imshow(new_input)
        # plt.subplot(1, 4, 4)
        # plt.imshow(new_output)
        plt.show()

    a = cv2.imread('../DIR-D/training/input/0.jpg')
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    cv2.imshow('a', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

