import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv2
from random import randint
import numpy as np


def change_brightness(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.adjust_brightness(image, delta=randint(20, 100))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    image = sess.run(image)
    return image/255


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

    res = change_brightness(img)
    plt.imshow(res)
    plt.show()
