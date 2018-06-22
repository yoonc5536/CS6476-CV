"""
CS6476: Problem Set 4 Tests

"""

import numpy as np
import cv2
import unittest
import ps4

INPUT_DIR = "input_images/test_images/"


class Part1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.input_imgs_1 = ['test_lk1.png', 'test_lk3.png', 'test_lk5.png']
        self.input_imgs_2 = ['test_lk2.png', 'test_lk4.png', 'test_lk6.png']

        self.delta_c = [0, 0, -1]
        self.delta_r = [0, -1, -1]

        self.r_val = [14, 12, 14]
        self.c_val = [15, 16, 15]

        self.cb = [(28, 30), (24, 32), (28, 30)]

        self.k_size = 15
        self.k_type = 'uniform'

    def test_optic_flow_LK(self):

        for i in range(3):

            f1 = self.input_imgs_1[i]
            f2 = self.input_imgs_2[i]

            img1 = cv2.imread(INPUT_DIR + f1, 0) / 255.
            img2 = cv2.imread(INPUT_DIR + f2, 0) / 255.

            u, v = ps4.optic_flow_lk(img1.copy(), img2.copy(),
                                     self.k_size, self.k_type, 1.)

            r = self.r_val[i]
            c = self.c_val[i]

            d_c = self.delta_c[i]
            d_r = self.delta_r[i]

            center_box = self.cb[i]

            u_mean = np.mean(u[r:r + center_box[0],
                             c:c + center_box[1]])

            check_u = abs(u_mean - d_c) <= 0.5

            error_msg = "Average of U values in the area where there is " \
                        "movement is greater than the allowed amount."

            self.assertTrue(check_u, error_msg)

            v_mean = np.mean(v[r:r + center_box[0],
                             c:c + center_box[1]])

            check_v = abs(v_mean - d_r) <= 0.5

            error_msg = "Average of V values in the area where there is " \
                        "movement is greater than the allowed amount."

            self.assertTrue(check_v, error_msg)


class Part2(unittest.TestCase):

    def test_reduce(self):
        input_imgs = ['test_reduce1_img.npy', 'test_reduce2_img.npy',
                      'test_reduce3_img.npy']
        ref_imgs = ['test_reduce1_ref.npy', 'test_reduce2_ref.npy',
                    'test_reduce3_ref.npy']

        for i in range(3):
            f1 = input_imgs[i]
            f2 = ref_imgs[i]

            test_array = np.load(INPUT_DIR + f1)

            reduced = ps4.reduce_image(test_array.copy())

            ref_reduced = np.load(INPUT_DIR + f2)

            correct = np.allclose(reduced, ref_reduced, atol=0.05)

            self.assertTrue(correct, "Output does not match the reference "
                                     "solution.")

    def test_expand(self):
        input_imgs = ['test_expand1_img.npy', 'test_expand2_img.npy',
                      'test_expand3_img.npy']
        ref_imgs = ['test_expand1_ref.npy', 'test_expand2_ref.npy',
                    'test_expand3_ref.npy']

        for i in range(3):
            f1 = input_imgs[i]
            f2 = ref_imgs[i]

            test_array = np.load(INPUT_DIR + f1)

            expanded = ps4.expand_image(test_array.copy())

            ref_expanded = np.load(INPUT_DIR + f2)

            correct = np.allclose(expanded, ref_expanded, atol=0.05)

            self.assertTrue(correct, "Output does not match the reference "
                                     "solution.")

    def test_gaussian_pyramid(self):
        input_imgs = ['test_gauss1_pyr.npy', 'test_gauss2_pyr.npy',
                      'test_gauss3_pyr.npy']
        ref_imgs = ['test_gauss1_pyr_ref.npy', 'test_gauss2_pyr_ref.npy',
                    'test_gauss3_pyr_ref.npy']
        levels = [4, 2, 4]

        for i in range(3):
            f1 = input_imgs[i]
            f2 = ref_imgs[i]
            l = levels[i]

            test_array = np.load(INPUT_DIR + f1)

            g_pyr = ps4.gaussian_pyramid(test_array.copy(), levels=l)

            g_pyr_ref = np.load(INPUT_DIR + f2)

            for l in range(len(g_pyr)):
                correct = np.allclose(g_pyr[l], g_pyr_ref[l], atol=0.1)

                error_msg = "Value at level {} does not match the answer." \
                            "".format(l)

                self.assertTrue(correct, error_msg)

    def test_laplacian_pyramid(self):
        input_imgs = ['test_lapl1_pyr.npy', 'test_lapl2_pyr.npy',
                      'test_lapl3_pyr.npy']
        ref_imgs = ['test_lapl1_pyr_ref.npy', 'test_lapl2_pyr_ref.npy',
                    'test_lapl3_pyr_ref.npy']
        levels = [5, 5, 4]

        for i in range(3):
            f1 = input_imgs[i]
            f2 = ref_imgs[i]

            test_array = np.load(INPUT_DIR + f1)

            l_pyr = ps4.laplacian_pyramid(test_array)

            l_pyr_ref = np.load(INPUT_DIR + f2)

            for l in range(levels[i]):
                correct = np.allclose(l_pyr[l], l_pyr_ref[l], atol=0.1)

                error_msg = "Value at level {} does not match the answer. " \
                            "Make sure your expand() function is passing " \
                            "the autograder.\n".format(l)

                self.assertTrue(correct, error_msg)


class Part3(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.input_imgs_1 = ['test_warp1.npy', 'test_warp3.npy',
                             'test_warp5.npy']
        self.input_imgs_2 = ['test_warp2.npy', 'test_warp4.npy',
                             'test_warp6.npy']
        self.input_flows = ['u_v1.npy', 'u_v2.npy', 'u_v3.npy']

        self.r_val = [6, 5, 8]
        self.c_val = [9, 8, 7]

        self.bv = [168, 139, 242]

    def test_warp(self):

        for i in range(2):
            f1 = self.input_imgs_1[i]  # Not used
            f2 = self.input_imgs_2[i]
            f3 = self.input_flows[i]

            img1 = np.load(INPUT_DIR + f1)  # Not used
            img2 = np.load(INPUT_DIR + f2)
            u_v = np.load(INPUT_DIR + f3)

            u = u_v[:, :, 0]
            v = u_v[:, :, 1]

            warped = ps4.warp(img2.copy(), u.copy(), v.copy(),
                              cv2.INTER_CUBIC, cv2.BORDER_REFLECT101)

            r = self.r_val[i]
            c = self.c_val[i]

            box_value = self.bv[i]

            center_box_average = np.mean(warped[r:3 * r, c:3 * c])
            correct_center_box = abs(center_box_average - box_value) <= 0.51

            error_msg = "Center box average pixel value is greater than the " \
                        "value used in the input image."

            self.assertTrue(correct_center_box, error_msg)

            warped_without_center = np.copy(warped)
            warped_without_center[r:3 * r, c:3 * c] = 0.

            average_warped_img = np.mean(warped_without_center)
            center_box_average = box_value * 0.15
            correct_warped_img = center_box_average >= average_warped_img

            error_msg = "Average of values outside the center box area are " \
                        "greater than the allowed amount."

            self.assertTrue(correct_warped_img, error_msg)


class Part4(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.input_imgs_1 = ['test_hlk1.png', 'test_hlk3.png', 'test_hlk5.png']

        self.input_imgs_2 = ['test_hlk2.png', 'test_hlk4.png', 'test_hlk6.png']

        self.delta_c = [-7, -1, 1]
        self.delta_r = [2, 6, 5]

        self.r_val = [17, 17, 16]
        self.c_val = [13, 17, 18]

        self.cb = [(34, 26), (34, 34), (32, 36)]

        self.k_size = 15
        self.k_type = 'uniform'

    def test_optic_flow_HLK(self):

        for i in range(3):

            f1 = self.input_imgs_1[i]
            f2 = self.input_imgs_2[i]

            img1 = cv2.imread(INPUT_DIR + f1, 0) / 255.
            img2 = cv2.imread(INPUT_DIR + f2, 0) / 255.

            u, v = ps4.hierarchical_lk(img1.copy(), img2.copy(), 3,
                                       self.k_size, self.k_type, 1.,
                                       cv2.INTER_CUBIC, cv2.BORDER_REFLECT101)

            r = self.r_val[i]
            c = self.c_val[i]

            d_c = self.delta_c[i]
            d_r = self.delta_r[i]

            center_box = self.cb[i]

            u_mean = np.mean(u[r:r + center_box[0],
                             c:c + center_box[1]])

            max_diff = abs(d_c) * .1 + .2
            check_u = abs(u_mean - d_c) < max_diff

            error_msg = "Average of U values in the area where there is " \
                        "movement is greater than the allowed amount."

            self.assertTrue(check_u, error_msg)

            v_mean = np.mean(v[r:r + center_box[0],
                             c:c + center_box[1]])

            max_diff = abs(d_r) * .1 + .2
            check_v = abs(v_mean - d_r) < max_diff

            error_msg = "Average of V values in the area where there is " \
                        "movement is greater than the allowed amount."

            self.assertTrue(check_v, error_msg)

if __name__ == "__main__":
    unittest.main()