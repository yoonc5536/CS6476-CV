"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to
                             [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    sobelx = cv2.Sobel(image,ddepth=-1,dx=1,dy=0,scale=1.0/8, ksize=3)
    return sobelx
    # raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    sobely = cv2.Sobel(image,ddepth=-1,dx=0,dy=1,scale=1.0/8, ksize=3)
    return sobely


    # raise NotImplementedError


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """


    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    It = img_b - img_a



    if k_type == 'uniform':
        kernel = np.ones((k_size,k_size),np.float32)/(k_size*k_size)
        a = cv2.filter2D(Ix*Ix,-1,kernel)
        b = cv2.filter2D(Ix*Iy,-1,kernel)
        c = cv2.filter2D(Ix*Iy,-1,kernel)
        d = cv2.filter2D(Iy*Iy,-1,kernel)
        e = cv2.filter2D(Ix*It,-1,kernel)
        f = cv2.filter2D(Iy*It,-1,kernel)

    if k_type == 'gaussian':
        a = cv2.GaussianBlur(Ix*Ix,(k_size,k_size),sigmaX=sigma, sigmaY=sigma)
        b = cv2.GaussianBlur(Ix*Iy,(k_size,k_size),sigmaX=sigma, sigmaY=sigma)
        c = cv2.GaussianBlur(Ix*Iy,(k_size,k_size),sigmaX=sigma, sigmaY=sigma)
        d = cv2.GaussianBlur(Iy*Iy,(k_size,k_size),sigmaX=sigma, sigmaY=sigma)
        e = cv2.GaussianBlur(Ix*It,(k_size,k_size),sigmaX=sigma, sigmaY=sigma)
        f = cv2.GaussianBlur(Iy*It,(k_size,k_size),sigmaX=sigma, sigmaY=sigma)

    noise = np.asscalar(0.001*np.random.rand(1, 1))
    denominator = a*d - b*c
    denominator[denominator == 0] = noise
    # denominant[np.where(denominant == 0)] = noise
    determinant = 1.0/denominator

    # x = d*e-b*f
    # y = -c*e+a*f
    # X = np.array([x,y])
    # solved_array = determinant * X
    # u = solved_array[0]
    # v = solved_array[1]
    # the elements of inverse Matrix of A

    e1 = determinant*d
    e2 = determinant*(-b)
    e3 = determinant*(-c)
    e4 = determinant*a

    # multiple the inverse Matrix of A with [e,
    # u = e4 * -e + e3 * -f
    # v = e2 * -f + e1 * -e

    u = e1 * -e + e3 * -f
    v = e2 * -e + e4 * -f



    # u = ((a * determinant * -e)) + ((-c * determinant)* -f)
    # v = ((determinant*(-b) * -f)) + ((determinant*d)* -e)
    #
    result = (u, v)
    # x = d*e-b*f
    # y = -c*e+a*f
    #
    # u = determinant * x
    # v = determinant * y

    # A=np.array([[a,b],[c,d]])
    #
    # A_abs=np.linalg.det(A)
    # A_inv = np.linalg.inv(A)
    # A = np.matrix([[a, b],[c, d]])
    # A = []
    # A = A.append([a, b])
    # A = A.append([c, d])


    # B = np.matrix([-e,-f])

    # noise = np.asscalar(0.00001*np.random.rand(1, 1))
    # A[np.where(A == 0)] = noise

    # A_inv = np.linalg.inv(A)

    # result = A_inv * B

    return result









    # raise NotImplementedError


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    parameter = 0.4
    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                     0.25, 0.25 - parameter /2.0])
    window = np.outer(kernel, kernel)

    blur_img = cv2.filter2D(image,-1,window)

    reduce_img = blur_img[::2,::2]

    return reduce_img



    # raise NotImplementedError


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    gaussianout = [0] * levels
    gaussianout[0] = image
    subsample = image
    for i in range(1,levels):
        subsample = reduce_image(subsample)
        gaussianout[i] = subsample

    return gaussianout
    # raise NotImplementedError


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    size_of_list = len(img_list)
    h_list = [0]*size_of_list
    w_list = [0]*size_of_list
    w_sum = 0
    width_sum_list = [0]*size_of_list
    for i in range(size_of_list):
        h = img_list[i].shape[0]
        w = img_list[i].shape[1]
        h_list[i]=h
        w_list[i]=w
        w_sum = w_sum + w
        width_sum_list[i] = w_sum
    max_h = np.max(h_list)

    #create array big enought to hold all the images
    new_img_base = np.zeros((max_h, w_sum), np.float32)


    for i in range(size_of_list):
        current_img = normalize_and_scale(img_list[i], scale_range=(0, 255))

        if i == 0:
            new_img_base[:h_list[i], :width_sum_list[i]] = current_img
        else:
            new_img_base[:h_list[i], width_sum_list[i-1]:width_sum_list[i]]=current_img

        # cv2.imshow("ddd", new_img_base)
        # cv2.waitKey(0)

    return new_img_base

    # raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    rows = image.shape[0]
    cols = image.shape[1]

    expand_image = np.zeros((rows*2,cols*2))

    expand_image[::2,::2] = image

    kernel = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
    expand_image = cv2.sepFilter2D(expand_image, -1, kernel, kernel)
    expand_image = 4.0*expand_image

    return expand_image


    # raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    size = len(g_pyr)
    l_pyr = [0]*size
    for i in range(size-1):
        tmp = expand_image(g_pyr[i+1])
        rows = g_pyr[i].shape[0]
        cols = g_pyr[i].shape[1]
        tmp = tmp[:rows, :cols]
        l_pyr[i] = g_pyr[i] - tmp
    l_pyr[-1] = g_pyr[-1]
    return l_pyr

    # raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    M, N = image.shape
    X, Y = np.meshgrid(xrange(N), xrange(M))
    X = X.astype('float32')
    Y = Y.astype('float32')
    U = U.astype('float32')
    V = V.astype('float32')
    X = X + U
    Y = Y + V

    wraped = cv2.remap(src=image,map1=X, map2=Y, interpolation=interpolation, borderMode=border_mode)

    # warped = np.zeros((rows,cols))
    return wraped

    # raise NotImplementedError


def match_size(img_a, img_b):

    rows_a, cols_a = img_a.shape[0],img_a.shape[1]

    rows_b, cols_b = img_b.shape[0],img_b.shape[1]

    min_rows = min(rows_a, rows_b)
    min_cols = min(cols_a, cols_b)


    img_a = img_a[:min_rows, :min_cols]
    img_b = img_b[:min_rows, :min_cols]
    return img_a, img_b

def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    pyr_img_a = gaussian_pyramid(img_a, levels)
    pyr_img_b = gaussian_pyramid(img_b, levels)

    half_smallest_pyr_l = reduce_image(pyr_img_a[-1])
    half_smallest_pyr_k = reduce_image(pyr_img_b[-1])

    cur_lel = levels
    while cur_lel > 0:
        if cur_lel == levels:
            u = np.zeros(half_smallest_pyr_l.shape)
            v = np.zeros(half_smallest_pyr_k.shape)

        u = 2*expand_image(u)
        v = 2*expand_image(v)

        a_k = pyr_img_a[cur_lel-1]
        b_k = pyr_img_b[cur_lel-1]

        u, a_k = match_size(u, a_k)
        v, b_k = match_size(v, b_k)

        w_k = warp(b_k, u, v, interpolation, border_mode)
        dx, dy = optic_flow_lk(a_k, w_k, k_size, k_type, sigma)
        #update u, and v
        u = u + dx
        v = v + dy
        cur_lel -= 1

    return (u,v)
    # raise NotImplementedError


