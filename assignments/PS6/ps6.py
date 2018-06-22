"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier



# A=np.array([1,2,3,4,5])
# B=np.array([2,3,3,4,6])
#
# inx = np.subtract(A, B)!=0.0

# M= A-np.array(A.mean(0),ndmin=2)
# S2,V2=np.linalg.eigh(np.dot(M.T,M))
#
# a = np.array([[1,], [2j, 5], [2j, 5]])
#
# w, v = np.linalg.eigh(a)
#
# xx = v[:, 0]


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   ([int]): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]

    x = []
    y = []
    for img in images_files:
        frame = cv2.imread(os.path.join(folder, img))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray,size, interpolation = cv2.INTER_CUBIC)
        flatten = res.flatten()
        x.append(flatten)
        name = img.split(".")
        for i in name[0]:
            if i.isdigit() and i != '0':
                cc = int(i)
        y.append(cc)

    x = np.asarray(x)
    y = np.asarray(y)

    result = (x, y)
    return result

    # raise NotImplementedError


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    cut = int(p*X.shape[0])
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:cut], indices[cut:]
    Xtrain = X[training_idx,:]
    ytrain = y[training_idx]
    Xtest = X[test_idx,:]
    ytest = y[test_idx]
    return (Xtrain,ytrain,Xtest,ytest)

    # raise NotImplementedError


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    x_mean =  np.mean(x, axis=0)

    return x_mean
    # raise NotImplementedError


def pca(X, k):
    """PCA Reduction method.
    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    M= X-np.array(X.mean(0),ndmin=2)

    eigenValues, eigenVectors = np.linalg.eigh(np.dot(M.T,M))

    # idx = eigenValues.argsort()[::-1]
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]
    eigenValues = eigenValues[::-1]
    eigenVectors = eigenVectors.T[::-1]
    eigenvalues = eigenValues[:k]
    eigenvectors = eigenVectors[:k].T

    return (eigenvectors, eigenvalues)

    # raise NotImplementedError


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    # def train(self):
    #     """Implement the for loop shown in the problem set instructions."""
    #     weights = np.ones(self.num_obs) / self.num_obs
    #     for i in range(self.num_iterations):
    #
    #         # Get predications
    #         h = WeakClassifier(self.Xtrain, self.ytrain, weights)
    #         h.train()
    #         predicated = h.predict(self.Xtrain)
    #         # find cumulative weights where prediction not equal to train
    #         idx = np.subtract(predicated, self.ytrain)!=0.0
    #         error = np.sum(weights[idx])
    #         alpha = 1/2.0*np.log((1.0-error)/(error+1e-15))
    #         self.alphas.append(alpha)
    #         self.weakClassifiers.append(h)
    #         if error > self.eps:
    #             a = -1*self.ytrain*alpha
    #             b = np.multiply((-1*self.ytrain*alpha,predicated))
    #             #based on the correct or incorrect, the incorrect gets more weights than the correct
    #             weightsfactor = np.exp(np.multiply((-1*self.ytrain*alpha,predicated)))
    #             weights = weights*weightsfactor
    #             # renormalize the weights
    #             sum_weights = np.sum(weights)
    #             weights = weights/sum_weights
    #         else:
    #             break
    #         # raise NotImplementedError

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        weights = np.ones(self.num_obs) / self.num_obs
        for i in range(self.num_iterations):


            # Get predications
            h = WeakClassifier(self.Xtrain, self.ytrain, weights)
            h.train()
            # error = sum([weights[img_idx] if self.ytrain[img_idx]!= h.predict(self.Xtrain[img_idx]) else 0 for img_idx in range(len(self.ytrain))])

            error = sum( map(lambda img_idx: weights[img_idx] if self.ytrain[img_idx]!= h.predict(self.Xtrain[img_idx]) else 0, range(len(self.ytrain))) )
            # predicated = h.predict(self.Xtrain)
            # # find cumulative weights where prediction not equal to train
            # idx = np.subtract(predicated, self.ytrain)!=0.0
            # error = np.sum(weights[idx])
            alpha = 0.5*np.log((1.0-error)/(error+1e-15))
            # alpha2 = 0.5*np.log((1.0-error)/error)
            self.alphas.append(alpha)

            self.weakClassifiers.append(h)
            if error > self.eps:

                weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.exp((-1*self.ytrain[img_idx]*alpha*h.predict(self.Xtrain[img_idx]))), range(len(weights)))))

                sum_weights = np.sum(weights)
                weights = weights/sum_weights

                # print weights
            #
            #     a = -1*self.ytrain*alpha
            #     b = np.multiply((-1*self.ytrain*alpha,predicated))
            #     #based on the correct or incorrect, the incorrect gets more weights than the correct
            #     weightsfactor = np.exp(np.multiply((-1*self.ytrain*alpha,predicated)))
            #     weights = weights*weightsfactor
                # renormalize the weights

            else:
                break
            # raise NotImplementedError

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        predicated = self.predict(self.Xtrain)
        difference = np.subtract(predicated, self.ytrain)
        correct = (difference == 0.0).sum()
        incorrect = (difference != 0.0).sum()

        return (correct, incorrect)

        # raise NotImplementedError

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        values = []
        for i in range(len(self.alphas)):
            predicates=[]
            for j in range(X.shape[0]):
                x = X[j,:]
                predicated = self.weakClassifiers[i].predict(x)
                predicates.append(predicated)
            predicates = np.asarray(predicates)
            values.append(self.alphas[i]*predicates)

        sumvalues = np.sum(values, axis=0)
        res = np.sign(sumvalues)
        res = np.asarray(res.copy())

        return res
        # raise NotImplementedError


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (x, y) position of the feature's top left corner.
        size (tuple): Feature's (width, height)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        
        img = np.zeros(shape)

        y1, x1 = self.position

        h,w = self.size

        strip = int(h/2)

        #left half is white, right half is gray
        img[y1:y1+strip, x1:x1+w] = 255
        img[y1+strip:y1+h, x1:x1+w] = 126
        return img

        # raise NotImplementedError

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        #initiate the image with all zeros size with the shape
        img = np.zeros(shape)

        y1, x1 = self.position

        h,w = self.size
        strip = int(w/2)

        #left half is white, right half is gray
        img[y1:y1+h, x1:x1+strip] = 255
        img[y1:y1+h, x1+strip:x1+w] = 126
        return img
        # raise NotImplementedError

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        
        img = np.zeros(shape)

        y1, x1 = self.position

        h,w = self.size
        
        strip = int(h/3)

        #left right 1/3 is white, mid 1/3 is gray
        img[y1:y1+strip, x1:x1+w] = 255
        img[y1+strip:y1+strip+strip, x1:x1+w] = 126
        img[y1+strip+strip:y1+h, x1:x1+w] = 255

        return img

        # raise NotImplementedError

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        
        img = np.zeros(shape)

        y1, x1 = self.position

        h,w = self.size
        
        strip = int(w/3)

        #left right 1/3 is white, mid 1/3 is gray
        img[y1:y1+h, x1:x1+strip] = 255
        img[y1:y1+h, x1+strip:x1+strip+strip] = 126
        img[y1:y1+h, x1+strip+strip:x1+w] = 255
        return img

        # raise NotImplementedError

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)

        y1, x1 = self.position

        h,w = self.size

        s1 = int(h/2)
        s2 = int(w/2)

        #left right 1/3 is white, mid 1/3 is gray
        img[y1:y1+s1, x1:x1+s2] = 126
        img[y1:y1+s1, x1+s2:x1+w] = 255
        img[y1+s1:y1+h, x1:x1+s2] = 255
        img[y1+s1:y1+h, x1+s2:x1+w] = 126

        return img


        # raise NotImplementedError

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (x, y) and (width, height) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """


        h,w = self.size
        ii = ii.astype(np.float32)

        if self.feat_type == (2, 1):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = self.position[0]+int(h/2)-1
            x2 = x1+w-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]+int(h/2)
            x1=self.position[1]
            y2=self.position[0]+int(h)-1
            x2=x1+w-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            score = A-B


        if self.feat_type == (1, 2):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = y1+int(h)-1
            x2 = x1+int(w/2)-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]
            x1=self.position[1]+int(w/2)
            y2=y1+int(h)-1
            x2=self.position[1]+int(w)-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            score = A-B


        #left right 1/3 is white, mid 1/3 is gray

        if self.feat_type == (3, 1):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = self.position[0]+int(h/3)-1
            x2 = x1+w-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]+int(h/3)
            x1=self.position[1]
            y2=self.position[0]+int(h/3)+int(h/3)-1
            x2=x1+w-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]+int(h/3)+int(h/3)
            x1=self.position[1]
            y2=self.position[0]+h-1
            x2=x1+w-1
            C = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            score = A-B+C


        if self.feat_type == (1, 3):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = y1+int(h)-1
            x2 = x1+int(w/3)-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]
            x1=self.position[1]+int(w/3)
            y2=y1+int(h)-1
            x2=self.position[1]+int(w/3)+int(w/3)-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]
            x1=self.position[1]+int(w/3)+int(w/3)
            y2=self.position[0]+h-1
            x2=self.position[1]+int(w)-1
            C = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            score = A-B+C


        if self.feat_type == (2, 2):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = y1+int(h/2)-1
            x2 = x1+int(w/2)-1
            A = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]
            x1=self.position[1]+int(w/2)
            y2=y1+int(h/2)-1
            x2=self.position[1]+w-1
            B = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]+int(h/2)
            x1=self.position[1]
            y2=self.position[0]+h-1
            x2=x1+int(w/2)-1
            C = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]

            y1=self.position[0]+int(h/2)
            x1=self.position[1]+int(w/2)
            y2=self.position[0]+h-1
            x2=self.position[1]+w-1
            D = ii[y2,x2]-ii[y1-1,x2]-ii[y2,x1-1]+ii[y1-1,x1-1]
            score = -A+B+C-D

        return score
        # raise NotImplementedError


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    res = []
    for i in range(len(images)):
        image = images[i]
        #cumsum over the row
        image = np.cumsum(image, axis=0)
        #cumsum over the
        image = np.cumsum(image, axis=1)
        res.append(image)

    return res

    # raise NotImplementedError


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.iteritems():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures



    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print " -- compute all scores --"
        for i, im in enumerate(self.integralImages):
            #using the haarfeatures to obtain the score for each image per each feature
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print " -- select classifiers --"
        for i in range(num_classifiers):
            # TODO: Complete the Viola Jones algorithm
            #Normalize the weights
            sum_weights = np.sum(weights)
            weights = weights/sum_weights

            #Instantiate a classifier hj
            h = VJ_Classifier(scores, self.labels, weights)
            #Train the classifier
            h.train()
            #for every feature
            # for i in range(len(self.haarFeatures)):
            #
            #     #map the function with the sequence
            #    error = sum( map(lambda img_idx: weights[img_idx] if self.labels[img_idx]!= h.predict(scores[img_idx]) else 0, range(len(self.integralImages))) )
            #the classifer already been fixed onced trained, error is the lowest error for found feature
            error = h.error
            #append hj to the self.classifiers
            self.classifiers.append(h)
            #update the weights
            beta = error/(1-error)
            weights = np.array(list(map(lambda img_idx: weights[img_idx] * beta if self.labels[img_idx] != h.predict(scores[img_idx]) else weights[img_idx] * 1.0, range(len(self.integralImages)))))
            #calculate the alpha
            alpha = np.log(1.0/beta)
            self.alphas.append(alpha)


            # raise NotImplementedError

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature
        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'
        for i, im in enumerate(ii):
            for clf in self.classifiers:
                f_idx = clf.feature
                feature = self.haarFeatures[f_idx]
                xx = [feature.evaluate(im)]
                scores[i, f_idx] = feature.evaluate(im)


        # for i, im in enumerate(ii):
        #     scores[i, :] = [self.haarFeatures[clf.feature].evaluate(im) for clf in self.classifiers]

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).

        for x in scores:
            # TODO
            # for alpha, clf in zip(self.alphas, self.classifiers):
            #     for alpha, clf in

            left = sum(map(lambda idx: self.alphas[idx]*self.classifiers[idx].predict(x), range(len(self.classifiers))))
            right = sum(map(lambda idx: 1/2*self.alphas[idx], range(len(self.alphas))))

            if left>=right:
                res = 1
            else:
                res = -1
            result.append(res)

            # raise NotImplementedError

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        frame = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define the window size
        windowsize_r = 24
        windowsize_c = 24

        # Crop out the window using the step size listed above
        x = []
        points1 = []
        points2 = []
        for r in range(0,gray.shape[0] - windowsize_r, 1):
            for c in range(0,gray.shape[1] - windowsize_c,1):
                points1.append([c, r])
                points2.append([c+windowsize_c, r+windowsize_r])
                window = gray[r:r+windowsize_r,c:c+windowsize_c]
                #flatten the window
                # flatten = window.flatten()
                x.append(np.array(window))
        #convert the list to array
        # x = np.asarray(x)

        # predications will be array that is 1/-1 for all the subimages
        predictions = self.predict(x)

        #To draw the rectangle, needs two points and also the draw area
        selectedp1s = []
        selectedp2s = []
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                point1 = points1[i]
                point2 = points2[i]
                selectedp1s.append(point1)
                selectedp2s.append(point2)

        selectedp1s = np.asarray(selectedp1s)
        selectedp2s = np.asarray(selectedp2s)

        p1 = np.mean(selectedp1s, axis=0).astype(np.int)
        p2 = np.mean(selectedp2s, axis=0).astype(np.int)

        cv2.rectangle(frame, tuple(p1), tuple(p2), (0,255,255), 2)
        # cv2.imshow('ss', frame)
        # cv2.waitKey(0)

        # now save the frame
        output_dir = "output"
        filename = filename + '.png'

        outputpath =os.path.join(output_dir, filename)
        cv2.imwrite(outputpath, frame)

        # raise NotImplementedError


