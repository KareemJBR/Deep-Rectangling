# import cv2
#
# image = cv2.imread("grass.jpg")
# image_norm = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# image_norm2 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
# image_norm3 = cv2.rotate(image, cv2.ROTATE_180)
#
# cv2.imshow('original Image', image)
# cv2.imshow('Rotated Image', image_norm)
# cv2.imshow('Rotated Image2', image_norm2)
# cv2.imshow('Rotated Image3', image_norm3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ----------------------------------------- trying different rotation degrees ---------------------------------------- #

# from PIL import Image
#
# original_image = Image.open("grass.jpg")
# rotated_image = original_image.rotate(40)
# rotated_image.save("rotated_image40.jpg")
# rotated_image = original_image.rotate(-40)
# rotated_image.save("rotated_image-40.jpg")
# # rotated_image.show()

# ----------------------------------------- testing seam carving ---------------------------------------- #

import cv2
import seam_carving

image = cv2.imread('tower.png')
res = seam_carving.resize(image, (137, 186))
res = seam_carving.resize(res, (200, 186))
cv2.imshow('original', image)
cv2.imshow('resized', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
