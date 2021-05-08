import cv2 as cv
import numpy as np

img = cv.imread(
    "D:/Disparity-Map-Generator/PSNR_Assignment2/3.0 output/disp3.png", cv.IMREAD_GRAYSCALE)

# img = cv.fastNlMeansDenoising(img, None, 15, 8, 15)
img = cv.blur(img, (15, 15))

img = cv.normalize(img, img, alpha=255,
                   beta=0, norm_type=cv.NORM_MINMAX)
img = np.uint8(img)
cv.imwrite(
    "D:/Disparity-Map-Generator/PSNR_Assignment2/PSNR_Python/pred/Art/disp1.png", img)
