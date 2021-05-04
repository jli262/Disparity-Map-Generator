import numpy as np
import cv2
from matplotlib import pyplot as plt
import imageio

imgL1 = cv2.imread(
    r'D:\Disparity-Map-Generator\StereoMatchingTestings\Art\view1.png', 0)
imgR1 = cv2.imread(
    r'D:\Disparity-Map-Generator\StereoMatchingTestings\Art\view5.png', 0)
imgL2 = cv2.imread(
    r'D:\Disparity-Map-Generator\StereoMatchingTestings\Dolls\view1.png', 0)
imgR2 = cv2.imread(
    r'D:\Disparity-Map-Generator\StereoMatchingTestings\Dolls\view5.png', 0)
imgL3 = cv2.imread(
    r'D:\Disparity-Map-Generator\StereoMatchingTestings\Reindeer\view1.png', 0)
imgR3 = cv2.imread(
    r'D:\Disparity-Map-Generator\StereoMatchingTestings\Reindeer\view5.png', 0)

stereo = cv2.StereoBM_create(numDisparities=240, blockSize=17)
disparity = stereo.compute(imgL1, imgR1)
disparity = cv2.normalize(disparity, disparity, alpha=255,
                         beta=0, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)
plt.imshow(disparity, 'gray')
imageio.imwrite(
    r'D:\Disparity-Map-Generator\PSNR_Assignment2\PSNR_Python\pred\Art\disp1.png', disparity)
# cv2.imwrite(r'D:\4186assign2\PSNR_Assignment2\PSNR_Python\pred\Art\disp1.png',disparity)

stereo = cv2.StereoBM_create(numDisparities=240, blockSize=17)
disparity = stereo.compute(imgL2, imgR2)
disparity = cv2.normalize(disparity, disparity, alpha=255,
                         beta=0, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)
plt.imshow(disparity, 'gray')
imageio.imwrite(
    r'D:\Disparity-Map-Generator\PSNR_Assignment2\PSNR_Python\pred\Dolls\disp1.png', disparity)
# cv2.imwrite(r'D:\4186assign2\PSNR_Assignment2\PSNR_Python\pred\Dolls\disp1.png',disparity)

stereo = cv2.StereoBM_create(numDisparities=240, blockSize=17)
disparity = stereo.compute(imgL3, imgR3)
disparity = cv2.normalize(disparity, disparity, alpha=255,
                         beta=0, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)
plt.imshow(disparity, 'gray')
imageio.imwrite(
    r'D:\Disparity-Map-Generator\PSNR_Assignment2\PSNR_Python\pred\Reindeer\disp1.png', disparity)
# cv2.imwrite(r'D:\4186assign2\PSNR_Assignment2\PSNR_Python\pred\Reindeer\disp1.png',disparity)
