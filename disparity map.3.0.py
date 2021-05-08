from matplotlib import pyplot as plt
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

def add_padding(input, padding):
    rows = input.shape[0]
    columns = input.shape[1]
    output = np.zeros((rows + padding * 2, columns + padding * 2), dtype=float)
    output[ padding : rows + padding, padding : columns + padding] = input
    return output

def search_bounds(column, block_size, width, rshift):
    disparity_range = 75
    padding = block_size // 2
    right_bound = column
    if rshift:
        left_bound = column - disparity_range
        if left_bound < padding:
            left_bound = padding
        step = 1
    else:
        left_bound = column + disparity_range
        if left_bound >= (width - 2*padding):
            left_bound = width - 2*padding - 2
        step = -1
    return left_bound, right_bound, step

grayscale_max = 255

l = cv2.imread('D:/Disparity-Map-Generator/StereoMatchingTestings/Art/view1.png', cv2.IMREAD_GRAYSCALE)
r = cv2.imread('D:/Disparity-Map-Generator/StereoMatchingTestings/Art/view5.png', cv2.IMREAD_GRAYSCALE)

block_size = 5

padding = block_size // 2
left_img = add_padding(l, padding)
right_img = add_padding(r, padding)

height, width = left_img.shape

# d_map = np.zeros((height - padding*2, width - padding*2), dtype=float)
d_map = np.zeros(l.shape, dtype=float)

rshift = True

for row in range(height - block_size + 1):
    for col in range(width - block_size + 1):

        bestdist = float('inf')
        shift = 0
        left_pixel = left_img[row:row + block_size, col:col + block_size]
        l_bound, r_bound, step = search_bounds(col, block_size, width, rshift)

        # for i in range(l_bound, r_bound - padding*2):
        for i in range(l_bound, r_bound, step):
            right_pixel = right_img[row:row + block_size, i:i + block_size]

            # if euclid_dist(left_pixel, right_pixel) < bestdist :
            ssd = np.sum((left_pixel - right_pixel) ** 2)
            # print('row:',row,' col:',col,' i:',i,' bestdist:',bestdist,' shift:',shift,' ssd:',ssd)
            if ssd < bestdist:
                bestdist = ssd
                shift = i

        if rshift:
            d_map[row, col] = col - shift
        else:
            d_map[row, col] = shift - col

disparity_SGBM = cv2.normalize(d_map, d_map, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(d_map)
cv2.imwrite("pred/Art/disp1.png", disparity_SGBM)
            
