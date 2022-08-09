import cv2.cv2 as cv2
import numpy as np


def cropped_image(_img, x1, x2, y1, y2):
    return _img[x1:x2, y1:y2]


def flip_image(_img):
    return np.fliplr(_img)


if __name__ == "__main__":
    img = cv2.imread("training/gt/00001.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # res, lab = augment(img, 'test')
    # plt.imshow(res)
    # plt.show()

    # res = blur_image(img)
    # plt.imshow(res)
    # plt.show()

    # res = cropped_image(img, 20, 500, 0, 200)
    # plt.imshow(res)
    # plt.show()
