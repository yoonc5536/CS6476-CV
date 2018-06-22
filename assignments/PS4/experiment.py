"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np
import ps4

# I/O directories
input_dir = "input_images"
output_dir = "output"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    cur_level = level

    expand_u = u
    expand_v = v

    u_list = [0] * (level+1)
    v_list = [0] * (level+1)
    u_list[cur_level] = u
    v_list[cur_level] = v


    while cur_level > 0:


        rows = expand_u.shape[0]
        cols = expand_v.shape[1]


        #update the expand_u, expand_v
        expand_u = np.zeros((rows*2,cols*2))
        expand_v = np.zeros((rows*2,cols*2))

        #not sure if this is correct or not???
        expand_u[::2,::2] = u_list[cur_level]
        expand_v[::2,::2] = v_list[cur_level]

        #aligh value

        expand_u = 2 * expand_u
        expand_v = 2 * expand_v

        #adjust to the reference pyramid row and col
        cur_level = cur_level -1
        pyramid_row = pyr[cur_level].shape[0]
        pyramid_col = pyr[cur_level].shape[1]
        expand_u = expand_u[:pyramid_row, :pyramid_col]
        expand_v = expand_v[:pyramid_row, :pyramid_col]

        #store the expand_u and expand_v


        u_list[cur_level] = expand_u
        v_list[cur_level] = expand_v

    return (expand_u, expand_v)




    # TODO: Your code here
    # raise NotImplementedError


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 40  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 32  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.


    # cv2.imshow('shift_0', shift_0)
    # cv2.waitKey(0)


    # Optional: smooth the images if LK doesn't work well on raw images
    {'k_size': 166, 'window': 'Param Tuning', 'sigma_img': 3, 'k_gauss_img': 17, 'k_type_val': 1, 'sigma': 0}

    img_a = cv2.GaussianBlur(shift_0,(17,17),3)
    img_b = cv2.GaussianBlur(shift_r10,(17,17),3)
    k_size = 167  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(img_a, img_b, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=2, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    # Now let's try with ###. You may want to try smoothing the
    # input images first.
    {'k_size': 89, 'window': 'Param Tuning', 'sigma_img': 14, 'k_gauss_img': 13, 'k_type_val': 0, 'sigma': 0}
    img_a = cv2.GaussianBlur(shift_0,(13,13),14)
    img_b = cv2.GaussianBlur(shift_r20,(13,13),14)
    k_size = 89  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(img_a, img_b, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=4, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)


    {'k_size': 106, 'window': 'Param Tuning', 'sigma_img': 14, 'k_gauss_img': 19, 'k_type_val': 0, 'sigma': 2}
    # Optional: smooth the images if LK doesn't work well on raw images
    img_a = cv2.GaussianBlur(shift_0,(19,19),14)
    img_b = cv2.GaussianBlur(shift_r40,(19,19),14)
    k_size = 107  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(img_a, img_b, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=4, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)

    # raise NotImplementedError


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 3  # TODO: Select the level number (or id) you wish to use
    k_size = 40  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_WRAP  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 3  # TODO: Select the level number (or id) you wish to use
    k_size = 40  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 3  # TODO: Define the number of levels
    k_size = 42  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    {'k_size': 68, 'window': 'Param Tuning', 'sigma_img': 0, 'k_gauss_img': 8, 'k_type_val': 0, 'sigma': 0}

    img_a = cv2.GaussianBlur(shift_0,(9,9),3)
    img_b = cv2.GaussianBlur(shift_r20,(9,9),3)
    k_size = 68
    k_type = "uniform"
    u20, v20 = ps4.hierarchical_lk(img_a, img_b, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    {'k_size': 89, 'window': 'Param Tuning', 'sigma_img': 11, 'k_gauss_img': 17, 'k_type_val': 0, 'sigma': 0}

    img_a = cv2.GaussianBlur(shift_0,(17,17),11)
    img_b = cv2.GaussianBlur(shift_r40,(17,17),11)
    k_size = 89
    k_type = "uniform"
    u40, v40 = ps4.hierarchical_lk(img_a, img_b, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    {'k_size': 42, 'window': 'Param Tuning', 'sigma_img': 2, 'k_gauss_img': 6, 'k_type_val': 0, 'sigma': 0}
    urban_img_01 = cv2.GaussianBlur(urban_img_01,(7,7),2)
    urban_img_02 = cv2.GaussianBlur(urban_img_02,(7,7),2)
    levels = 4  # TODO: Define the number of levels
    k_size = 42  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def frame_interpolation(img_t00, img_t10, levels, k_size, k_type, sigma,
                        interpolation_lk, border_mode_lk, interpolation_warp, border_mode_warp):


    #find the displacement matrix u, v
    u, v = ps4.hierarchical_lk(img_t00, img_t10, levels, k_size,
                               k_type, sigma, interpolation_lk, border_mode_lk)

    img_t02 = ps4.warp(img_t00, -0.2*u, -0.2*v, interpolation_warp, border_mode_warp)

    img_t04 = ps4.warp(img_t02, -0.4*u, -0.4*v, interpolation_warp, border_mode_warp)
    img_t06 = ps4.warp(img_t04, -0.6*u, -0.6*v, interpolation_warp, border_mode_warp)
    img_t08 = ps4.warp(img_t06, -0.8*u, -0.8*v, interpolation_warp, border_mode_warp)

    img_t_p = ps4.warp(img_t10, u, v, interpolation_warp, border_mode_warp)

    row1 = np.hstack([img_t00,img_t02,img_t04])
    row2 = np.hstack([img_t06,img_t08,img_t10])
    collage = np.vstack([row1, row2])
    # cv2.imshow("aa", collage)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(output_dir, "500.png"),  ps4.normalize_and_scale(img_t00))
    cv2.imwrite(os.path.join(output_dir, "502.png"),  ps4.normalize_and_scale(img_t02))
    cv2.imwrite(os.path.join(output_dir, "504.png"), ps4.normalize_and_scale(img_t04))
    cv2.imwrite(os.path.join(output_dir, "506.png"), ps4.normalize_and_scale(img_t06))
    cv2.imwrite(os.path.join(output_dir, "508.png"), ps4.normalize_and_scale(img_t08))
    cv2.imwrite(os.path.join(output_dir, "510.png"), ps4.normalize_and_scale(img_t10))
    cv2.imwrite(os.path.join(output_dir, "510_p.png"), ps4.normalize_and_scale(img_t_p))
    return collage



def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    levels = 4  # TODO: Define the number of levels
    k_size = 10  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation_lk = cv2.INTER_CUBIC  # You may try different values
    border_mode_lk = cv2.BORDER_REFLECT101  # You may try different values
    interpolation_warp = cv2.INTER_CUBIC  # You may try different values
    border_mode_warp = cv2.BORDER_REFLECT101  # You may try different values

    img_t00 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    img_t10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),0) / 255.

    collage1 = frame_interpolation(img_t00, img_t10, levels, k_size, k_type, sigma,
                                   interpolation_lk, border_mode_lk, interpolation_warp, border_mode_warp)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-1.png"), ps4.normalize_and_scale(collage1))



    # raise NotImplementedError


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    levels = 4  # TODO: Define the number of levels
    k_size = 80  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation_lk = cv2.INTER_CUBIC  # You may try different values
    border_mode_lk = cv2.BORDER_REFLECT101  # You may try different values
    interpolation_warp = cv2.INTER_CUBIC  # You may try different values
    border_mode_warp = cv2.BORDER_REFLECT101  # You may try different values

    img_t00 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc01.png'), 0) / 255.
    img_t10 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc02.png'), 0) / 255.

    collage1 = frame_interpolation(img_t00, img_t10, levels, k_size, k_type, sigma,
                                   interpolation_lk, border_mode_lk, interpolation_warp, border_mode_warp)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-b-1.png"), ps4.normalize_and_scale(collage1))


    levels = 4  # TODO: Define the number of levels
    k_size = 80  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation_lk = cv2.INTER_CUBIC  # You may try different values
    border_mode_lk = cv2.BORDER_REFLECT101  # You may try different values
    interpolation_warp = cv2.INTER_CUBIC  # You may try different values
    border_mode_warp = cv2.BORDER_REFLECT101  # You may try different values


    img_t00 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc02.png'), 0) / 255.
    img_t10 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc03.png'), 0) / 255.


    collage2 = frame_interpolation(img_t00, img_t10, levels, k_size, k_type, sigma,
                                   interpolation_lk, border_mode_lk, interpolation_warp, border_mode_warp)
    # cv2.imshow("chk", collage2)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-b-2.png"), ps4.normalize_and_scale(collage2))

    # raise NotImplementedError

def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)
    # video = videoCapture.open(filename)
    # chk  = video.open()

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None

def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    # fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

OUT_DIR = "output"
def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)

def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    video_name = "ps4-my-video.mp4"
    fps = 40
    VID_DIR = "input_videos"
    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)
    frame_list = []
    image = image_gen.next()
    h, w, d = image.shape
    out_path = "output/ps4-6-output.mp4"
    #initiate the video output
    video_out = mp4_video_writer(out_path, (w, h), fps)
    frame_num = 1
    output_counter = 1
    while image is not None:

        print "Processing fame {}".format(frame_num)
        frame_list.append(image)

        if frame_num == 1:
            pass
        else:
            image_src = frame_list[-2]
            image_input1 = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)/255.
            image_input2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
            k_size = 80  # TODO: Select a kernel size
            k_type = "uniform"  # TODO: Select a kernel type
            sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
            u, v = ps4.optic_flow_lk(image_input1, image_input2, k_size, k_type, sigma)

            # Flow image
            image_out = image_src.copy()
            scale=3
            stride=10
            color=(0, 255, 0)
            # if frame_num == 843:
            for y in xrange(0, v.shape[0], stride):
                    for x in xrange(0, u.shape[1], stride):
                        cv2.line(image_out, (x, y), (x + int(u[y, x] * scale),
                                                   y + int(v[y, x] * scale)), color, 1)
                        cv2.circle(image_out, (x + int(u[y, x] * scale),
                                             y + int(v[y, x] * scale)), 1, color, 1)

            output_prefix = 'ps4-6-a'

            frame_ids = [1, 250]


            frame_id = frame_ids[(output_counter - 1) % 2]

            if (frame_num-1) == frame_id:
                out_str = output_prefix + "-{}.png".format(output_counter)
                save_image(out_str, image_out)
                output_counter += 1

            video_out.write(image_out)



        image = image_gen.next()
        frame_num += 1

    video_out.release()




    # raise NotImplementedError


if __name__ == "__main__":
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
