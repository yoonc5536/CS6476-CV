"""
CS6476: Problem Set 3 Tests

In this script you will find some simple tests that will help you
determine if your implementation satisfies the autograder
requirements. In this collection of tests your code output will be
tested to verify if the correct data type is returned. Additionally,
there are a couple of examples with sample answers to guide you better
in developing your algorithms.
"""

import numpy as np
import cv2
import unittest
import ps3

INPUT_DIR = "input_images/test_images/"


def ssim(x, y, k1, k2):
    """Calculates the SSIM index from two uint8 images

    Source: eq. 13 from Wang et. al.:
    Image Quality Assessment: From Error Visibility to Structural Similarity

    Args:
        x (numpy.array): Non-negative uint8 image
        y (numpy.array): Non-negative uint8 image
        k1 (float): constant << 1 to avoid instability when ux^2 + uy^2 is
                    very close to zero.
        k2 (float): constant << 1 to avoid instability when sx^2 + sy^2 is
                    very close to zero.

    Returns:
        float: Structural Similarity Index
    """
    x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    y_gray = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

    ux = np.mean(x_gray)
    uy = np.mean(y_gray)

    x_c = x_gray.copy() - ux
    y_c = y_gray.copy() - uy

    sx = np.std(x_gray)
    sy = np.std(y_gray)

    n = x_gray.size
    sxy = np.sum(x_c * y_c) / (n - 1)

    c1 = 255 * k1
    c2 = 255 * k2

    num = (2 * ux * uy + c1) * (2 * sxy + c2)
    den = (ux ** 2 + uy ** 2 + c1) * (sx ** 2 + sy ** 2 + c2)

    return num / den


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.height, self.width = height, width = 200, 500  # Rows, Cols
        self.corner_positions = [(0, 0), (0, height - 1), (width - 1, 0),
                                 (width - 1, height - 1)]

        # Creates a gradient color image for testing
        y = np.linspace(1, 255, height)
        x = np.linspace(1, 255, width)
        a, b = np.meshgrid(x, y)
        self.test_grad_image = cv2.merge((a, b, b[::-1])).astype('uint8')
        self.test_grad_image.setflags(write=False)  # Copies will be writeable

        # Test markers.
        # Marker order: [top-left (x,y), bottom-left, top-right, bottom-right]
        self.marker_positions = [(145, 54), (154, 150), (360, 70), (340, 130)]

        # Homography for testing
        self.homography = cv2.getPerspectiveTransform(
            np.array(self.corner_positions, 'float32'),
            np.array(self.marker_positions, 'float32'))
        self.homography.setflags(write=False)  # Copies will be writeable

    def test_get_corners_list(self):
        clist = ps3.get_corners_list(np.zeros((self.height, self.width)))
        self.assertEqual(len(clist), 4, msg="List should have a length of 4.")

        lengthError = "Each item in the marker list should have length of 2."
        self.assertEqual(len(clist[0]), 2, msg=lengthError)

        ValueIndexError = "%s value at index %d off by %f"

        for i, (a, b) in enumerate(zip(clist, self.corner_positions)):
            self.assertEqual(a[0], b[0],
                             msg=ValueIndexError % ('X', i, abs(a[0] - b[0])))
            self.assertEqual(a[1], b[1],
                             msg=ValueIndexError % ('Y', i, abs(a[1] - b[1])))

    def test_find_markers_simple_rectangle(self):

        file_names = ['simple_rectangle.png', 'simple_rectangle_noisy.png',
                      'simple_rectangle_noisy_gaussian.png']

        template = cv2.imread("input_images/template.jpg")

        markers_pos = [(100, 40), (100, 150), (400, 40), (400, 150)]

        thresh = 1

        for f in file_names:

            test_image = cv2.imread(INPUT_DIR + f)

            ret_markers = ps3.find_markers(test_image, template)

            for act_pt, ret_pt in zip(markers_pos, ret_markers):

                x_dist_test = abs(ret_pt[0] - act_pt[0]) <= thresh
                self.assertTrue(x_dist_test,
                                msg='X point is too far from reference. '
                                    'Expected: {}. Returned: {}'
                                    ''.format(ret_pt[0], act_pt[0]))

                y_dist_test = abs(ret_pt[1] - act_pt[1]) <= thresh
                self.assertTrue(y_dist_test,
                                msg='Y point is too far from reference. '
                                    'Expected: {}. Returned: {}'
                                    ''.format(ret_pt[1], act_pt[1]))

    def test_find_markers_wall_image(self):

        file_names = ['rectangle_wall.png', 'rectangle_wall_noisy.png',
                      'rectangle_wall_noisy_gaussian.png']

        template = cv2.imread("input_images/template.jpg")

        markers_pos = [(197, 288), (283, 640), (979, 99), (1062, 465)]

        thresh = 1

        for f in file_names:

            test_image = cv2.imread(INPUT_DIR + f)

            ret_markers = ps3.find_markers(test_image, template)

            for act_pt, ret_pt in zip(markers_pos, ret_markers):

                x_dist_test = abs(ret_pt[0] - act_pt[0]) <= thresh
                self.assertTrue(x_dist_test,
                                msg='X point is too far from reference. '
                                    'Expected: {}. Returned: {}'
                                    ''.format(ret_pt[0], act_pt[0]))

                y_dist_test = abs(ret_pt[1] - act_pt[1]) <= thresh
                self.assertTrue(y_dist_test,
                                msg='Y point is too far from reference. '
                                    'Expected: {}. Returned: {}'
                                    ''.format(ret_pt[1], act_pt[1]))

    def test_solving_for_homography(self):
        # Gets the student's result
        homography = ps3.find_four_point_transform(self.corner_positions,
                                                   self.marker_positions)

        self.assertEqual(homography.shape, (3, 3),
                         msg="Homography should be a 3 by 3 array.")

        self.assertTrue(
            homography.dtype not in ('uint8', 'int', 'int8', 'int16', 'int32'),
            msg="Homography data type should be float")

        self.assertTrue(0.999 < homography[2, 2] < 1.001,
                        msg="Bottom right of homography should be 1")

        # Using "[0]" to change from 3D to 2D arrays of points
        student_warped_corners = \
            cv2.perspectiveTransform(
                np.array([self.corner_positions], 'float32'),
                homography)[0]

        cv2_warped_corners = \
            cv2.perspectiveTransform(
                np.array([self.corner_positions], 'float32'),
                self.homography)[0]

        acceptable_offset = 1.0

        error_msg = "Warped points too far from expected position."

        for a, b in zip(student_warped_corners, cv2_warped_corners):
            self.assertTrue(
                np.linalg.norm(np.array(a) - np.array(b)) <= acceptable_offset,
                msg=error_msg)

    def test_projecting_image(self):
        black_bg = np.zeros_like(self.test_grad_image)

        # Get the student's result
        ret_image = ps3.project_imageA_onto_imageB(self.test_grad_image.copy(),
                                                   black_bg, self.homography)

        err_msg = "Output array shape should be same shape " \
                  "as 2nd parameter image."

        self.assertEqual(ret_image.shape, black_bg.shape, msg=err_msg)

        self.assertEqual(ret_image.dtype, black_bg.dtype, msg=err_msg)

        # Creates a comparison final image
        comparison_image = cv2.warpPerspective(self.test_grad_image,
                                               self.homography,
                                               (self.width, self.height))

        # Test the entire image total value difference
        diff = ssim(ret_image, comparison_image, 1e-3, 1e-3)

        acceptable_ssim = 0.95

        error_msg = "Warp is too far from expected result (diff={})".format(
            diff)

        self.assertTrue(diff >= acceptable_ssim, msg=error_msg)


if __name__ == "__main__":
    unittest.main()