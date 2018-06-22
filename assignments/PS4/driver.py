#USAGE Example:
# Tune_Motion_In_Image could be any function which accepts constants (e.g. images) and variables (variable names are keys of dictionary as shown below)
# Example Signature (can have any variable(s), but **make sure** to include 'window=None' in signature (the last argument as shown below)):
# def Tune_Motion_In_Image(img_a, img_b, outfile, k_size=11, k_type_val=1, sigma=5, k_gauss_img=3, sigma_img=2, window=None):
#       ... #(your logic)
#
#       #**MUST** include following 2 lines in your desired function
#       windowName = 'result' if window is None else window
#       cv2.imshow(windowName, <output image you want to display with trackbars>)
#
#       ...

import sys
import os
sys.path.append('C:\Users\zz\Dropbox (Partners HealthCare)\CS6476CV\assignments')
from assignments.utility import *
import ps4
import experiment
sys.setrecursionlimit(100000000)

# I/O directories
input_dir = "input_images"
output_dir = "output"

def Tune_Motion_In_Image(img_a, img_b, outfile, k_size=11, k_type_val=1, sigma=5, k_gauss_img=3, sigma_img=2, window=None):

    k_type="uniform" if k_type_val<0.5 else "gaussian"
    k_gauss_img += (k_gauss_img + 1) % 2

    kernel = (k_gauss_img,k_gauss_img)

    img_a = cv2.GaussianBlur(img_a,kernel,sigma_img)
    img_b = cv2.GaussianBlur(img_b,kernel,sigma_img)

    # img_a = cv2.blur(img_a,kernel)
    # img_b = cv2.blur(img_b,kernel)
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    levels = 4
    u10, v10 = ps4.hierarchical_lk(img_a, img_b, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    # u, v = ps4.optic_flow_lk(img_a, img_b, k_size, k_type, sigma)
    u_v = experiment.quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, outfile), u_v)

    windowName = 'result' if window is None else window
    cv2.imshow(windowName, u_v)



def Tune_Warp_In_Image(img_a, img_b, outfile, k_size=11, k_type_val=1, sigma=5, k_gauss_img=3, sigma_img=2, interpolation_type= 3, window=None):

    k_type="uniform" if k_type_val<0.5 else "gaussian"
    k_gauss_img += (k_gauss_img + 1) % 2

    k_size += (k_size + 1) % 2

    kernel = (k_gauss_img,k_gauss_img)

    img_a = cv2.GaussianBlur(img_a,kernel,sigma_img)
    img_b = cv2.GaussianBlur(img_b,kernel,sigma_img)


    u, v = ps4.optic_flow_lk(img_a, img_b, k_size, k_type, sigma)
    u, v = experiment.scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)


    if interpolation_type<0.5:
        interpolation = cv2.INTER_NEAREST
    elif interpolation_type<1.5:
        interpolation =cv2.INTER_LINEAR
    elif interpolation_type<2.5:
        interpolation =cv2.INTER_AREA
    elif interpolation_type<3.5:
        interpolation =cv2.INTER_CUBIC
    else:
        interpolation =cv2.INTER_LANCZOS4

    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, outfile),
                ps4.normalize_and_scale(diff_yos_img_01_02))

    windowName = 'result' if window is None else window
    cv2.imshow(windowName, diff_yos_img_01_02)


def Frame_Motion(img_a, img_b, outfile, k_size=11, k_type_val=1, sigma=5, k_gauss_img=3, sigma_img=2, window=None):

    k_type="uniform" if k_type_val<0.5 else "gaussian"
    k_gauss_img += (k_gauss_img + 1) % 2

    kernel = (k_gauss_img,k_gauss_img)

    img_a = cv2.GaussianBlur(img_a,kernel,sigma_img)
    img_b = cv2.GaussianBlur(img_b,kernel,sigma_img)

    interpolation_lk = cv2.INTER_CUBIC  # You may try different values
    border_mode_lk = cv2.BORDER_REFLECT101  # You may try different values
    interpolation_warp = cv2.INTER_CUBIC  # You may try different values
    border_mode_warp = cv2.BORDER_REFLECT101  # You may try different values

    levels = 4
    collage1 = experiment.frame_interpolation(img_a, img_b, levels, k_size, k_type, sigma,
                                   interpolation_lk, border_mode_lk, interpolation_warp, border_mode_warp)
    cv2.imwrite(os.path.join(output_dir, outfile), collage1)

    windowName = 'result' if window is None else window
    cv2.imshow(windowName, collage1)

if __name__ == '__main__':
    print "--- Problem Set 4 GUI ---"
    variables = {'k_size': 200,
                 'k_type_val': 1, # this is just a switch. I used inside my Tune_Motion_In_Image function as k_type="uniform" if k_type_val<0.5 else "gaussian"
                 'sigma': 20,
                 'k_gauss_img': 30,
                 'sigma_img': 20,
                }
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                          'ShiftR5U5.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.



    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    level_id = 3

    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    img_t00 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    img_t10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),0) / 255.

    img_t00 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc01.png'), 0) / 255.
    img_t10 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc03.png'), 0) / 255.




    tune_params = Render(Tune_Motion_In_Image, (shift_0, shift_r40, "ps4-1-b-3.png"), variables)
    tune_params.execute()