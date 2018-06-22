"""
CS6476: Problem Set 2 Tests

In this script you will find some simple tests that will help you
determine if your implementation satisfies the autograder
requirements. In this collection of tests your code output will be
tested to verify if the correct data type is returned. Additionally,
there are a couple of examples with sample answers to guide you better
in developing your algorithms.
"""

import cv2
import unittest
import ps2


def check_result(label, coords, ref, tol):
    assert (abs(coords[0] - ref[0]) <= tol and
            abs(coords[1] - ref[1]) <= tol), "Wrong coordinate values. " \
                                             "Image used: {}. " \
                                             "Expected: ({}, {}), " \
                                             "Returned: ({}, {}). " \
                                             "Max tolerance: {}." \
                                             "".format(label, ref[0], ref[1],
                                                       coords[0], coords[1],
                                                       tol)


class TestTrafficLight(unittest.TestCase):
    """Test Traffic Light Detection"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_output(self):
        test_image = cv2.imread("input_images/test_images/simple_tl_test.png")
        radii_range = range(10, 30, 1)
        result = ps2.traffic_light_detection(test_image, radii_range)

        self.assertTrue(result is not None, "Output is NoneType.")
        self.assertEqual(2, len(result), "Output should be a tuple of 2 "
                                         "elements.")

        coords = result[0]
        state = result[1]

        is_tuple = isinstance(coords, (tuple))
        self.assertTrue(is_tuple, "Coordinates output is not a tuple.")

        is_string = isinstance(state, str)
        self.assertTrue(is_string, "Traffic light state is not a string.")

        if state not in ["red", "yellow", "green"]:
            raise (ValueError, "Traffic light state is not valid.")

    def test_simple_tl(self):
        tl_images = {"simple_tl_test": {"x": 45, "y": 120, "state": "green"},
                     "tl_green_299_287_blank": {"x": 287, "y": 299,
                                                "state": "green"},
                     "tl_red_199_137_blank": {"x": 137, "y": 199,
                                              "state": "red"},
                     "tl_yellow_199_237_blank": {"x": 237, "y": 199,
                                                 "state": "yellow"}}

        radii_range = range(10, 30, 1)

        for tl in tl_images:
            tl_data = tl_images[tl]
            test_image = cv2.imread("input_images/test_images/"
                                    "{}.png".format(tl))

            coords, state = ps2.traffic_light_detection(test_image,
                                                        radii_range)

            check_result(tl, coords, (tl_data["x"], tl_data["y"]), 5)
            self.assertEqual(state, tl_data["state"], "Wrong state value.")

    def test_scene_tl(self):
        tl_images = {"scene_tl_test": {"x": 338, "y": 200, "state": "red"},
                     "tl_green_299_287_background": {"x": 287, "y": 299,
                                                     "state": "green"},
                     "tl_red_199_137_background": {"x": 137, "y": 199,
                                                   "state": "red"},
                     "tl_yellow_199_237_background": {"x": 237, "y": 199,
                                                      "state": "yellow"}}

        radii_range = range(10, 30, 1)

        for tl in tl_images:
            tl_data = tl_images[tl]
            test_image = cv2.imread("input_images/test_images/"
                                    "{}.png".format(tl))

            coords, state = ps2.traffic_light_detection(test_image,
                                                        radii_range)

            check_result(tl, coords, (tl_data["x"], tl_data["y"]), 5)
            self.assertEqual(state, tl_data["state"], "Wrong state value.")


class TestTrafficSignsBlank(unittest.TestCase):
    """Test Traffic Sign Detection using a blank background"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_stop_sign(self):
        image_name = "stop_249_149_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.stop_sign_detection(test_image)

        check_result(image_name, coords, (149, 249), 5)

    def test_construction_sign(self):
        image_name = "construction_150_200_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.construction_sign_detection(test_image)

        check_result("construction_150_200_blank", coords, (200, 150), 5)

    def test_warning_sign(self):
        image_name = "warning_250_300_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.warning_sign_detection(test_image)

        check_result(image_name, coords, (300, 250), 5)

    def test_do_not_enter_sign(self):
        image_name = "no_entry_145_145_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.do_not_enter_sign_detection(test_image)

        check_result(image_name, coords, (145, 145), 5)

    def test_yield_sign(self):
        image_name = "yield_173_358_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.yield_sign_detection(test_image)

        check_result(image_name, coords, (358, 173), 5)


class TestTrafficSignsScene(unittest.TestCase):
    """Test Traffic Sign Detection using a simulated street scene"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_stop_sign(self):
        image_name = "stop_249_149_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.stop_sign_detection(test_image)

        check_result(image_name, coords, (149, 249), 5)

    def test_construction_sign(self):
        image_name = "construction_150_200_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.construction_sign_detection(test_image)

        check_result("construction_150_200_blank", coords, (200, 150), 5)

    def test_warning_sign(self):
        image_name = "warning_250_300_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.warning_sign_detection(test_image)

        check_result(image_name, coords, (300, 250), 5)

    def test_do_not_enter_sign(self):
        image_name = "no_entry_145_145_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.do_not_enter_sign_detection(test_image)

        check_result(image_name, coords, (145, 145), 5)

    def test_yield_sign(self):
        image_name = "yield_173_358_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.yield_sign_detection(test_image)

        check_result(image_name, coords, (358, 173), 5)


if __name__ == "__main__":
    unittest.main()
