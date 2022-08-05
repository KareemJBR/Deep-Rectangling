import tensorflow as tf
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import numpy as np


def change_brightness(image, delta):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = (image / 255.0)
    image = tf.image.adjust_brightness(image, delta=delta)
    return image


def blur_image(_img):
    return cv2.GaussianBlur(_img, (5, 5), 0)


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

    res = change_brightness(img, 0)
    plt.imshow(res)
    plt.show()
