import cv2
import numpy as np

img1 = cv2.imread('Dharani_1.jpg', 1)
print(img1.shape)
img2 = cv2.imread('black_updated.png', 1)
print(img2.shape)
result = np.concatenate((img1, img2), axis=0)
cv2.imshow('merged_image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()