"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        # State vector (X) with the initial x and y values.,x
        self.state = np.matrix([[init_x], [init_y], [0.], [0.]])  # state

        # Covariance 4x4 array () initialized with a diagonal matrix with some value.
        self.I = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])

        # 4x4 state transition matrix Dt, F
        dt = 0.1
        self.Dt =  np.matrix([[1.,0.,dt,0],[0.,1.,0.,dt],[0.,0.,1.,0.],[0.,0.,0.,1.]])

        # 2x4 measurement matrix Mt
        # measurement function Mt
        self.Mt = np.matrix([[1.,0.,0.,0.],[0.,1.,0.,0.]])

        # 4x4 process noise matrix dt

        self.dt = Q

        # 2x2 measurement noise matrix mt, R

        self.mt =  R

        # P ....
        self.P =  np.matrix([[1000.,0.,0.,0.],[0.,1000.,0.,0.],[0.,0.,1000.,0.],[0.,0.,0.,1000.]])

        # raise NotImplementedError

    def predict(self):

        self.state = self.Dt * self.state
        self.P = self.Dt * self.P * self.Dt.transpose() + self.dt


        # raise NotImplementedError

    def correct(self, meas_x, meas_y):
        # Y are the measurements at time t
        Y = np.matrix([meas_x, meas_y])
        S = self.Mt * self.P * self.Mt.transpose() + self.mt
        # Kalman Gain K
        K = self.P * self.Mt.transpose() * np.linalg.inv(S)
        y = Y.transpose() - (self.Mt * self.state)
        self.state = self.state + (K * y)
        self.P = (self.I - (K * self.Mt)) * self.P

        # raise NotImplementedError

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)
        res1 = self.state[0,0]
        res2 = self.state[1,0]

        return res1, res2


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder, control std
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        self.alpha = 0


        #defines the image
        self.imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to rgb
        imageSize = np.array(self.imageGray.shape)
        self.imageSize = imageSize[::-1]

        minYCoord = int(self.template_rect['y'])
        maxYCoord = int(minYCoord + self.template_rect['h'])
        minXCoord = int(self.template_rect['x'])
        maxXCoord = int(minXCoord + self.template_rect['w'])
        model = self.imageGray[minYCoord:maxYCoord, minXCoord:maxXCoord]
        self.model = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.stateDims = kwargs.get('dism', 2)

        minBound = [minXCoord, minYCoord]
        maxBound = [maxXCoord, maxYCoord]

        # add the third dimension for the scale factor, initiallized as uniformly distributed
        if self.stateDims == 3:
            self.imageSize = np.append(self.imageSize, 1)
            minBound.append(0.9)
            maxBound.append(1.0)

        # draw same number of particles for x, and y by using uniform distribution from 0 to the width and height
        # Initialize your particles array. Read the docstring.
        self.particles = np.array([np.random.uniform(minBound[i], maxBound[i],self.num_particles) for i in range(self.stateDims)]).T
        self.weights = np.ones(len(self.particles)) / len(self.particles)  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.
        # this is the index of the particles, starts from 0 to 99 if 100 particles
        self.particleIdxs = np.arange(self.num_particles)

        # raise NotImplementedError

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    #this function is referenced at
    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        if np.subtract(template.shape, frame_cutout.shape).any():
            return 0.0
        else:
            mse = np.sum(np.subtract(template, frame_cutout, dtype=np.float32) ** 2)
            mse /= float(template.shape[0] * template.shape[1])
            prob = np.exp(-mse / 2 / self.sigma_exp**2)
            return prob

        # return NotImplementedError


    def particleInitial(self):
        # particleInitial particles using a normal distribution centered around 0, this sigma_dyn is the control std
        noise = np.random.normal(0, self.sigma_dyn, self.particles.shape)
        self.particles = self.particles + noise
        imageWidth, imageHeight = self.imageSize[:2]
        self.particles[:,0] = np.clip(self.particles[:,0], 0, imageWidth - 1)
        self.particles[:,1] = np.clip(self.particles[:,1], 0, imageHeight - 1)


    def particleInitialMD(self):
        # particleInitial particles using a normal distribution centered around 0, this sigma_dyn is the control std
        noise = np.zeros(self.particles.shape)
        noise[:,:2] = np.random.normal(0, self.sigma_dyn,(self.num_particles,2))
        noise[:,2] = np.random.normal(0, self.sigma_scale,self.num_particles)
        self.particles = self.particles + noise

        imageWidth, imageHeight = self.imageSize[:2]
        self.particles[:,0] = np.clip(self.particles[:,0], 0, imageWidth - 1)
        self.particles[:,1] = np.clip(self.particles[:,1], 0, imageHeight - 1)

        # clip scale in case the scale goes out of the limints between 0,1
        self.particles[:,2] = np.clip(self.particles[:,2], 0.2, 0.99)


    def compareParticle_Model(self, img):

        modelHeight, modelWidth = self.model.shape[:2]
        minXCoord = (self.particles[:,0] - modelWidth/2)
        minYCoord = (self.particles[:,1] - modelHeight/2)
        minXCoord = np.array([round(i,0) for i in minXCoord]).astype(np.int)
        minYCoord = np.array([round(i,0) for i in minYCoord]).astype(np.int)

        maxXCoord = minXCoord + modelWidth
        maxYCoord = minYCoord + modelHeight


        # maxXCoord = np.array([round(i,0) for i in maxXCoord]).astype(np.int)
        # maxYCoord = np.array([round(i,0) for i in maxYCoord]).astype(np.int)
        # construct a list contains all possible similar template created by particle filters
        particleTemplates = [0] * self.num_particles
        # get the particle arounding potential template
        for i in range(self.num_particles):
            temp = img[minYCoord[i]:maxYCoord[i], minXCoord[i]:maxXCoord[i]]
            particleTemplates[i] = temp
        # construct a list contains the weights, compute importance weight - similarity of each patch to the model
        self.weights = np.array([self.get_error_metric(self.model, particleTemplate) for particleTemplate in particleTemplates])
        # normalize the weights
        sum_weights = np.sum(self.weights)
        self.weights = self.weights/sum_weights


    def compareParticle_ModelMD(self, img):
        # get template corresponding to each particle
        modelHeight, modelWidth = self.model.shape[:2]
        minXCoord = (self.particles[:,0] - modelWidth/2).astype(np.int)
        minYCoord = (self.particles[:,1] - modelHeight/2).astype(np.int)
        # construct a list contains all possible similar template created by particle filters
        particleTemplates = []
        frame_cutouts = []
        # get the particle souranding potential template
        for i in range(self.num_particles):
            temp = img[minYCoord[i]:minYCoord[i]+modelHeight, minXCoord[i]:minXCoord[i]+modelWidth]
            s = self.particles[:,2][i]
            try:
                frame_cutout = cv2.resize(self.model.copy(), (0, 0), fx=s, fy=s)
                frame_cutouts.append(frame_cutout)
            except Exception, e:
                print str(e)

            if temp.size == 0:
                particleTemp = temp.copy()
            else:
                try:
                    particleTemp = cv2.resize(temp.copy(), (0, 0), fx=s, fy=s)
                except:
                    print "error"

            particleTemplates.append(particleTemp)
            weight = self.get_error_metric(particleTemp,frame_cutout)
            self.weights[i] = weight

        # construct a list contains the weights, compute importance weight - similarity of each patch to the model
        # self.weights = np.array([self.get_error_metric(particleTemplate, frame_cutout) for particleTemplate,frame_cutout in zip(particleTemplates,frame_cutouts)])

        # normalize the weights
        sum_weights = np.sum(self.weights)
        self.weights = self.weights/sum_weights



    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """

        imageWidth, imageHeight = self.imageSize[:2]
        # sample new particle indices using the probability distribution of the weights
        # the first argument is the pool, which is sequential numbers, the second argument is
        # the number of numbers returned, and p is the weights
        j = np.random.choice(self.particleIdxs, self.num_particles, True, p=self.weights.T)
        # resample the particles using the distribution of the weights
        self.particles = np.array(self.particles[j])

        # resample particles also need to add some noises


        if self.stateDims == 3:
            noise = np.zeros(self.particles.shape)
            noise[:,0] = np.random.normal(0, 5,self.num_particles)
            noise[:,1] = np.random.normal(0, 5,self.num_particles)
            noise[:,2] = 0
            scaleValue = np.mean(self.particles[:,2])
            # noise[:,2] = np.random.normal(0, self.sigma_scale,self.num_particles)
            # print "   scaleValue", scaleValue
            #add some penality to make sure the propotion is not zero
            self.scaleModel = cv2.resize(self.model.copy(), (0, 0), fx=scaleValue+0.001, fy=scaleValue+0.001)
            # self.scaleModel = self.model
            self.particles = self.particles + noise

        # # clip particles in case the particle goes out of the window limits

        bestIdx = cv2.minMaxLoc(self.weights)[3][1]
        bestState = self.particles[bestIdx]
        draw_area = self.model
        if self.stateDims == 3:
            draw_area = self.scaleModel
        point1 = (bestState[:2] - np.array(draw_area.shape[::-1])/2).astype(np.int)
        point2 = point1 + np.array(draw_area.shape[::-1])

        # self.particles[:,0] = np.clip(self.particles[:,0], point1[0], point2[0])
        # self.particles[:,1] = np.clip(self.particles[:,1], point1[1], point2[1])
        # clip scale in case the scale goes out of the limints between 0,1
        if self.stateDims == 3:
            self.particles[:,2] = np.clip(self.particles[:,2], 0.1, 0.99)

        # return NotImplementedError


    def returnBestParticle(self):

        # modelHeight, modelWidth = self.model.shape[:2]
        # minXCoord = (self.particles[:,0] - modelWidth/2).astype(np.int)
        # minYCoord = (self.particles[:,1] - modelHeight/2).astype(np.int)
        # construct a list contains all possible similar template created by re-sampled particles
        # particleTemplates = [0] * self.num_particles
        # # get the particle arounding potential template
        # for i in range(self.num_particles):
        #     temp = self.imageGray[minYCoord[i]:minYCoord[i]+modelHeight, minXCoord[i]:minXCoord[i]+modelWidth]
        #     particleTemplates[i] = temp
        # threshold = np.array([self.get_error_metric(particleTemplate, self.model) for particleTemplate in particleTemplates])
        # print "     max is", np.max(threshold)
        # print "     min is", np.min(threshold)
        # print "     median is", np.median(threshold)
        stateIdx = np.random.choice(self.particleIdxs, 1, p=self.weights)
        # based on the resampled particles, only one state will be selected
        self.state = self.particles[stateIdx][0]

    def particleDecidedModel(self, frame):
        # get current model based on belief
        modelHeight, modelWidth = self.model.shape[:2]
        # the current best state is based on the particle filters

        minXCoord = (self.state[0] - modelWidth/2)
        minYCoord = (self.state[1] - modelHeight/2)
        minXCoord = int(round(minXCoord,0))
        minYCoord = int(round(minYCoord,0))

        maxXCoord = minXCoord + modelWidth
        maxYCoord = minYCoord + modelHeight

        bestModel = frame[minYCoord:maxYCoord, minXCoord:maxXCoord]

        # apply appearance model update if new model shape is unchanged
        if bestModel.shape == self.model.shape:
            self.model = self.alpha * bestModel + (1-self.alpha) * self.model
            self.model = self.model

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        self.imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.stateDims == 2:
            self.particleInitial()
            self.compareParticle_Model(self.imageGray)
        if self.stateDims == 3:
            self.particleInitialMD()
            self.compareParticle_ModelMD(self.imageGray)

        self.resample_particles()
        self.returnBestParticle()
        if self.alpha > 0:
            self.particleDecidedModel(self.imageGray)

        # raise NotImplementedError

    def visualize_filter(self, img):
        self.draw_particles(img)
        self.draw_window(img)
        self.draw_std(img)

    def draw_particles(self, img):
        for p in self.particles:
            cv2.circle(img, tuple(p.astype(int)), 2, (180,255,0), -1)

    def draw_window(self, img):
        bestIdx = cv2.minMaxLoc(self.weights)[3][1]
        bestState = self.particles[bestIdx]
        point1 = (bestState - np.array(self.model.shape[::-1])/2).astype(np.int)
        #  point1 = (self.state - np.array(self.model.shape[::-1])/2).astype(np.int)
        point2 = point1 + np.array(self.model.shape[::-1])
        cv2.rectangle(img, tuple(point1), tuple(point2), (0,255,0), 2)

    def draw_std(self, img):
        weighted_sum = 0
        dist = np.linalg.norm(self.particles - self.state)
        weighted_sum = np.sum(dist * self.weights.reshape((-1,1)))
        cv2.circle(img, tuple(self.state.astype(np.int)),int(weighted_sum), (255,255,255), 1)



    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        #draw particle
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, (x_weighted_mean.astype(int),y_weighted_mean.astype(int)), 2, (0,0,255), -1)


        for p in self.particles:
            p = p[:2]
            cv2.circle(frame_in, tuple(p.astype(int)), 2, (0,255,0), -1)


        #draw the window

        bestIdx = cv2.minMaxLoc(self.weights)[3][1]
        bestState = self.particles[bestIdx]
        draw_area = self.model
        if self.stateDims == 3:
            draw_area = self.scaleModel
        point1 = (bestState[:2] - np.array(draw_area.shape[::-1])/2).astype(np.int)
        #  point1 = (self.state - np.array(self.model.shape[::-1])/2).astype(np.int)
        point2 = point1 + np.array(draw_area.shape[::-1])
        cv2.rectangle(frame_in, tuple(point1), tuple(point2), (0,255,0), 2)

        #draw stand deviation of all the particles using the best one as the center
        weighted_sum = 0
        dist = np.linalg.norm(self.particles - self.state)
        weighted_sum = np.sum(dist * self.weights.reshape((-1,1)))
        try:
            cv2.circle(frame_in, tuple(self.state[:2].astype(np.int)),int(weighted_sum), (255,255,255), 2)
        except ValueError, e:
            print "value error" + str(e)


        # Complete the rest of the code as instructed.
        # raise NotImplementedError


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    # def process(self, frame):
    #     """Processes a video frame (image) and updates the filter's state.
    #
    #     This process is also inherited from ParticleFilter. Depending on your
    #     implementation, you may comment out this function and use helper
    #     methods that implement the "Appearance Model" procedure.
    #
    #     Args:
    #         frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].
    #
    #     Returns:
    #         None.
    #     """


        # raise NotImplementedError


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.sigma_scale = kwargs.get('sigma_scale', 0.01)

    # def process(self, frame):
    #     """Processes a video frame (image) and updates the filter's state.
    #
    #     This process is also inherited from ParticleFilter. Depending on your
    #     implementation, you may comment out this function and use helper
    #     methods that implement the "More Dynamics" procedure.
    #
    #     Args:
    #         frame (numpy.array): color BGR uint8 image of current video frame,
    #                              values in [0, 255].
    #
    #     Returns:
    #         None.
    #     """
        # raise NotImplementedError