import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read both images and convert to grayscale
img1 = cv.imread(r'D:\Disparity-Map-Generator\StereoMatchingTestings\Art\view1.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread(r'D:\Disparity-Map-Generator\StereoMatchingTestings\Art\view5.png', cv.IMREAD_GRAYSCALE)

# ------------------------------------------------------------
# PREPROCESSING

# Compare unprocessed images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1, cmap="gray")
axes[1].imshow(img2, cmap="gray")
axes[0].axvline(250)
axes[1].axvline(250)
axes[0].axvline(450)
axes[1].axvline(450)
plt.suptitle("Original images")
plt.show()