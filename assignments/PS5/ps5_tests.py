import numpy as np
import cv2
import unittest
import ps5
import os

#Older numpy versions can't deal with zero covariance
NOISE_0 = {'x': 0.0001, 'y': 0.0001}
NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def load_data(file_path):
    centers = np.load("%s/centers.npy"%file_path)
    images = np.zeros((100, 300, 300, 3))
    for i in xrange(100):
        images[i,:,:,:] = cv2.imread("%s/%d.jpg"%(file_path, i))
    return (images, centers)


def kf_matching_sensor(frame, noise, template):
    corr_map = cv2.matchTemplate(frame.astype(np.uint8), template,
                                 cv2.cv.CV_TM_SQDIFF)
    z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)
    z_w = template.shape[0]
    z_h = template.shape[1]
    z_x += z_w // 2 + np.random.normal(0, noise['x'])
    z_y += z_h // 2 + np.random.normal(0, noise['y'])
    return z_x, z_y


def get_centers_distance(pt_1, pt_2):
    return np.sqrt((float(pt_1[0]) - float(pt_2[0])) ** 2 + (float(pt_1[1]) - float(pt_2[1]))**2)


def assert_scene(kf, images, centers, noise, template, CheckDistance):
    for i in xrange(100):
        s_center = kf_matching_sensor(images[i,:,:,:], noise, template)
        k_center = kf.process(*s_center)
        distance = get_centers_distance(k_center, centers[i,:])
        CheckDistance(distance)


def get_predicted_center(pf):
    particles = pf.get_particles()
    weights = pf.get_weights()

    u_weighted_mean = 0
    v_weighted_mean = 0
    for i in range(pf.num_particles):
        u_weighted_mean += particles[i, 0] * weights[i]
        v_weighted_mean += particles[i, 1] * weights[i]

    return [u_weighted_mean, v_weighted_mean]


def visualize_filter(pf, frame):
    out_frame = frame.copy()
    pf.render(out_frame)
    cv2.imshow('test', out_frame)
    cv2.imshow('template', pf.template.astype(np.uint8))
    cv2.waitKey(1)


class PS5_KF_Tests(unittest.TestCase):

    def test_KF_Blank(self):
        file_path = "input_images/input_test/blank"
        template = cv2.imread("input_images/input_test/template.jpg")
        images, centers = load_data(file_path)
        kf = ps5.KalmanFilter(*centers[0,:])
        assert_scene(kf, images, centers, NOISE_0, template,
                     lambda x: self.assertAlmostEqual(x, 0, delta=10))

    def test_KF_Color(self):
        file_path = "input_images/input_test/color"
        template = cv2.imread("input_images/input_test/template.jpg")
        images, centers = load_data(file_path)
        kf = ps5.KalmanFilter(*centers[0,:])
        assert_scene(kf, images, centers, NOISE_0, template,
                     lambda x: self.assertAlmostEqual(x, 0, delta=15))

class PS5_PF_Tests(unittest.TestCase):
    # Using png files to improve resolution

    def get_scene_info(self, test_path, template_path):

        img_path = os.path.join(test_path, "images")
        img_list = [f for f in os.listdir(img_path) if f.endswith(".png")]
        img_list.sort()

        points_path = os.path.join(test_path, "points.npy")
        points_array = np.load(points_path)

        frame = cv2.imread(os.path.join(img_path, img_list[0]))
        template = cv2.imread(template_path)

        t_center = points_array[0]
        t_h, t_w = template.shape[:2]
        template_rect = {'x': t_center[0] - t_w // 2,
                         'y': t_center[1] - t_h // 2,
                         'w': t_w,
                         'h': t_h}

        return frame, template_rect, template, img_path, img_list, points_array

    def check_distance(self, distance, max_distance, frame_num,
                       predicted_center, actual_center):
        self.assertTrue(distance <= max_distance,
                        "Test failed at frame: {} \n"
                        "Estimated center is too far from the actual "
                        "center. \n"
                        "Student's center: {} \n"
                        "Actual center: {} \n"
                        "Max euclidean distance allowed: {}".format(frame_num,
                                                                    predicted_center,
                                                                    actual_center,
                                                                    max_distance))

    def run_filter(self, pf, img_path, img_list, points_array, shape_ref):

        for i, img in enumerate(img_list):
            frame = cv2.imread(os.path.join(img_path, img_list[i]))
            c_x, c_y = points_array[i, :]

            pf.process(frame)

            if True:  # Set to true if you want to see all frames
                visualize_filter(pf, frame)

            if i > 10:
                predicted_center = get_predicted_center(pf)
                distance = get_centers_distance([c_x, c_y], predicted_center)

                max_distance = max(shape_ref) * 1.25

                self.check_distance(distance, max_distance, i,
                                    predicted_center, (c_x, c_y))

    def test_PF_base_1(self):
        test_path = "input_images/input_test/circle/1/"
        template_path = "input_images/input_test/circle/template.png"

        scene = self.get_scene_info(test_path, template_path)

        frame, template_rect, template, img_path, img_list, points_array = scene

        num_particles = 100
        sigma_mse = 10.
        sigma_dyn = 10.
        pf = ps5.ParticleFilter(frame, template,
                                num_particles=num_particles,
                                sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                                template_coords=template_rect,
                                temp=template_rect)

        shape_radius = [25]
        self.run_filter(pf, img_path, img_list, points_array, shape_radius)

    def test_PF_base_2(self):
        test_path = "input_images/input_test/circle/2/"
        template_path = "input_images/input_test/circle/template.png"

        scene = self.get_scene_info(test_path, template_path)

        frame, template_rect, template, img_path, img_list, points_array = scene

        num_particles = 100
        sigma_mse = 10.
        sigma_dyn = 10.
        pf = ps5.ParticleFilter(frame, template,
                                num_particles=num_particles,
                                sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                                template_coords=template_rect,
                                temp=template_rect)

        shape_radius = [25]
        self.run_filter(pf, img_path, img_list, points_array, shape_radius)

    def test_PF_base_3(self):
        test_path = "input_images/input_test/circle/3/"
        template_path = "input_images/input_test/circle/template.png"

        scene = self.get_scene_info(test_path, template_path)

        frame, template_rect, template, img_path, img_list, points_array = scene

        num_particles = 100
        sigma_mse = 10.
        sigma_dyn = 10.
        pf = ps5.ParticleFilter(frame, template,
                                num_particles=num_particles,
                                sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                                template_coords=template_rect,
                                temp=template_rect)

        shape_radius = [25]
        self.run_filter(pf, img_path, img_list, points_array, shape_radius)

    def test_PF_ellipse_1(self):
        test_path = "input_images/input_test/ellipse/1/"
        template_path = "input_images/input_test/ellipse/template.png"

        scene = self.get_scene_info(test_path, template_path)

        frame, template_rect, template, img_path, img_list, points_array = scene

        pf = ps5.AppearanceModelPF(frame, template,
                                   num_particles=400,
                                   sigma_exp=10., sigma_dyn=20.,
                                   alpha=.05, template_coords=template_rect)

        axes_shape = (50, 25)
        self.run_filter(pf, img_path, img_list, points_array, axes_shape)

    def test_PF_ellipse_2(self):
        test_path = "input_images/input_test/ellipse/2/"
        template_path = "input_images/input_test/ellipse/template.png"

        scene = self.get_scene_info(test_path, template_path)

        frame, template_rect, template, img_path, img_list, points_array = scene

        pf = ps5.AppearanceModelPF(frame, template,
                                   num_particles=400,
                                   sigma_exp=10., sigma_dyn=20.,
                                   alpha=.05, template_coords=template_rect)

        axes_shape = (50, 25)
        self.run_filter(pf, img_path, img_list, points_array, axes_shape)

    def test_PF_ellipse_3(self):
        test_path = "input_images/input_test/ellipse/3/"
        template_path = "input_images/input_test/ellipse/template.png"

        scene = self.get_scene_info(test_path, template_path)

        frame, template_rect, template, img_path, img_list, points_array = scene

        pf = ps5.AppearanceModelPF(frame, template,
                                   num_particles=400,
                                   sigma_exp=10., sigma_dyn=20.,
                                   alpha=.05, template_coords=template_rect)

        axes_shape = (50, 25)
        self.run_filter(pf, img_path, img_list, points_array, axes_shape)


if __name__ == '__main__':
    unittest.main()

