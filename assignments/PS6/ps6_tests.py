import numpy as np
import unittest
import ps6
import os
import cv2
INPUT_DIR = "input_images/input_test"


class PCA(unittest.TestCase):

    def test_mean_face(self):

        for i in range(1, 4):
            file_name = "x_data_mean_{}.npy".format(i)
            file_path = os.path.join(INPUT_DIR, file_name)
            x_data = np.load(file_path)

            file_name = "correct_mean_{}.npy".format(i)
            file_path = os.path.join(INPUT_DIR, file_name)
            x_mean = np.load(file_path)

            result = ps6.get_mean_face(x_data)

            correct = np.allclose(result, x_mean, atol=1)
            message = "Values do not match the reference solution. " \
                      "This function should only compute the mean of each " \
                      "column."
            self.assertTrue(correct, message)

    def test_pca(self):

        k_list = [3, 3, 4]

        for i in range(1, 4):

            k = k_list[i-1]

            file_name = "x_data_pca_k{}_{}.npy".format(k, i)
            file_path = os.path.join(INPUT_DIR, file_name)
            x_data = np.load(file_path)

            file_name = "ref_val_pca_k{}_{}.npy".format(k, i)
            file_path = os.path.join(INPUT_DIR, file_name)
            ref_val = np.load(file_path)

            eig_vecs, eig_vals = ps6.pca(x_data, k)

            correct = eig_vals.size == k
            message = "Wrong number of eigenvalues. K used: {}. " \
                      "Returned eigenvalues: {}".format(k, eig_vals)
            self.assertTrue(correct, message)

            for j in range(k):
                correct = np.allclose(ref_val[j], eig_vals[j])
                message = "Wrong eigenvalue at position {}".format(i)
                self.assertTrue(correct, message)


class Boosting(unittest.TestCase):

    def setUp(self):
        self.threshold_list = [[87, 62, 122, 59, 125],
                               [138, 69, 54, 117, 61],
                               [65, 108,  95,  82, 132]]

        self.y_pred_ref = ["ypred_boosting_1.npy", "ypred_boosting_2.npy",
                           "ypred_boosting_3.npy"]

        self.x_data = np.load(os.path.join(INPUT_DIR, "x_boosting_1.npy"))
        self.alphas = np.load(os.path.join(INPUT_DIR, "alpha_1.npy"))

    def test_predict(self):
        thresh_vals_list = self.threshold_list
        y_pred_ref_list = self.y_pred_ref
        x_data = self.x_data
        alphas = self.alphas

        for thresh_vals, y_ref_file in zip(thresh_vals_list, y_pred_ref_list):
            y_pred_ref = np.load(os.path.join(INPUT_DIR, y_ref_file))

            # Create a list of untrained classifiers to verify
            # predict operation.
            classifiers = [ps6.WeakClassifier([], [], [], thresh=t) for t in
                           thresh_vals]

            ytrain = np.zeros((x_data.shape[0],))  # Doesn't matter for this test
            numIte = 0  # Doesn't matter for this test

            test = ps6.Boosting(x_data, ytrain, numIte)

            # Modifying object variables for testing
            test.weakClassifiers = classifiers
            test.alphas = alphas

            y_pred = test.predict(x_data)

            correct = np.allclose(y_pred, y_pred_ref)
            message = "Predictions do not match reference predictions."
            self.assertTrue(correct, message)

    def test_evaluate(self):
        thresh_vals_list = self.threshold_list

        x_data = self.x_data
        alphas = self.alphas

        corr_inc_list = [(277, 283), (279, 281), (274, 286)]
        y_data = np.load(os.path.join(INPUT_DIR, "y_boosting_1.npy"))

        iter_lists = zip(thresh_vals_list, corr_inc_list)
        for thresh_vals, corr_inc in iter_lists:

            # Create a list of untrained classifiers to verify
            # predict operation.
            classifiers = [ps6.WeakClassifier([], [], [], thresh=t) for t in
                           thresh_vals]

            ytrain = np.zeros((x_data.shape[0],))  # Doesn't matter for this test
            numIte = 0  # Doesn't matter for this test

            test = ps6.Boosting(x_data, ytrain, numIte)

            # Modifying object variables for testing
            test.weakClassifiers = classifiers
            test.alphas = alphas
            test.Xtrain = x_data
            test.ytrain = y_data

            correct, incorrect = test.evaluate()

            correct_test = np.allclose(correct, corr_inc[0])
            message = "Correct values do not match the reference."
            self.assertTrue(correct_test, message)

            incorrect_test = np.allclose(incorrect, corr_inc[1])
            message = "Incorrect values do not match the reference."
            self.assertTrue(incorrect_test, message)


class HaarFeature(unittest.TestCase):

    def setUp(self):
        self.input_dir = os.path.join("input_images", "input_test")

    def test_preview(self):
        feat_types = 2 * [(1, 2)] + 2 * [(2, 1)] + 2 * [(1, 3)] \
                     + 2 * [(3, 1)] + 2 * [(2, 2)]
        positions = 5 * [(6, 9), (8, 5)]
        sizes = 5 * [(14, 18), (11, 15)]
        samples = 5 * [0, 1]

        for j in range(10):
            feat_type = feat_types[j]
            pos = positions[j]
            size = sizes[j]
            s = samples[j]
            file_name = "haar_preview_ft{}_pos{}_sz{}_{}.npy".format(feat_type, pos, size, s)

            ref_img = np.load(os.path.join(self.input_dir, file_name))

            xx = ref_img[19,17]
            yy = ref_img[20,17]
            zz = ref_img[19,18]

            # Uncomment if you want to see the reference image
            # cv2.imshow("ref_img", ref_img.astype("uint8"))
            # cv2.waitKey(0)

            hf = ps6.HaarFeature(feat_type, pos, size)
            hf_img = hf.preview((50, 50))

            correct = np.allclose(hf_img, ref_img)
            message = "Output image does not match the reference solution."
            self.assertTrue(correct, message)

    def test_integral_images(self):
        ti_path = os.path.join(self.input_dir, "test_image_ii.npy")
        ii_path = os.path.join(self.input_dir, "integral_image_ii.npy")

        test_image = np.load(ti_path)
        integral_image = np.load(ii_path)

        r = np.random.randint(1, integral_image.shape[0])
        c = np.random.randint(1, integral_image.shape[1])

        result = ps6.convert_images_to_integral_images([test_image])

        ref_val = integral_image[r, c]

        correct = np.allclose(ref_val, result[0][r, c])
        message = "Value at row:{}, col:{} is not correct. "
        self.assertTrue(correct, message)

    def test_HaarFeature_evaluate(self):
        ti_path = os.path.join(self.input_dir, "test_image_ii.npy")
        ii_path = os.path.join(self.input_dir, "integral_image_ii.npy")

        test_image = np.load(ti_path)
        integral_image = np.load(ii_path)

        feat_type = (2, 2)  # Change feature type
        pos = (5, 5)
        size = (30, 30)

        if feat_type == (2, 1):
            A = np.sum(test_image[pos[0]:pos[0] + size[0] / 2,
                                  pos[1]:pos[1] + size[1]])
            B = np.sum(test_image[pos[0] + size[0] / 2:pos[0] + size[0],
                                  pos[1]:pos[1] + size[1]])
            ref = A - B

        if feat_type == (1, 2):
            A = np.sum(test_image[pos[0]:pos[0] + size[0],
                                  pos[1]:pos[1] + size[1] / 2])
            B = np.sum(test_image[pos[0]:pos[0] + size[0],
                                  pos[1] + size[1] / 2:pos[1] + size[1]])
            ref = A - B

        if feat_type == (3, 1):
            A = np.sum(test_image[pos[0]:pos[0] + size[0] / 3,
                                  pos[1]:pos[1] + size[1]])
            B = np.sum(test_image[pos[0] + size[0] / 3:pos[0] + 2 * size[0] / 3,
                                  pos[1]:pos[1] + size[1]])
            C = np.sum(test_image[pos[0] + 2 * size[0] / 3:pos[0] + size[0],
                                  pos[1]:pos[1] + size[1]])
            ref = A - B + C

        if feat_type == (1, 3):
            A = np.sum(test_image[pos[0]:pos[0] + size[0],
                                  pos[1]:pos[1] + size[1] / 3])
            B = np.sum(test_image[pos[0]:pos[0] + size[0],
                                  pos[1] + size[1] / 3:pos[1] + 2 * size[1] / 3])
            C = np.sum(test_image[pos[0]:pos[0] + size[0],
                                  pos[1] + 2 * size[1] / 3:pos[1] + size[1]])
            ref = A - B + C

        if feat_type == (2, 2):
            A = np.sum(test_image[pos[0]:pos[0] + size[0] / 2,
                                  pos[1]:pos[1] + size[1] / 2])
            B = np.sum(test_image[pos[0]:pos[0] + size[0] / 2,
                                  pos[1] + size[1] / 2:pos[1] + size[1]])
            C = np.sum(test_image[pos[0] + size[0] / 2:pos[0] + size[0],
                                  pos[1]:pos[1] + size[1] / 2])
            D = np.sum(test_image[pos[0] + size[0] / 2:pos[0] + size[0],
                                  pos[1] + size[1] / 2:pos[1] + size[1]])
            ref = -A + B + C - D

        hf = ps6.HaarFeature(feat_type, pos, size)

        score = hf.evaluate(integral_image)

        correct = np.allclose(score, ref)

        message = "Wrong score returned."
        self.assertTrue(correct, message)


if __name__ == '__main__':
    unittest.main()
