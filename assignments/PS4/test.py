import numpy as np
import cv2


k_size = 5

parameter = 0.4
kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                     0.25, 0.25 - parameter /2.0])
window = np.outer(kernel, kernel)

kernel2 = cv2.getGaussianKernel( 11, 1.7 )

loop = 100000
cur_loop = loop
while cur_loop > 0:
    noise = np.asscalar(0.001*np.random.rand(1, 1))
    if noise == 0:
        print noise
    cur_loop=cur_loop-1

import sys
print sys.maxsize

k_gauss_img = 9
k_gauss_img += (k_gauss_img + 1) % 2

chk = 4