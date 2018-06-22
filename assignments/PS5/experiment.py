"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import ps5
import os
import numpy as np

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


# Helper code
def run_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print 'Working on frame %d' % frame_num


def run_kalman_filter(kf, imgs_dir, noise, sensor, save_frames={},
                      template_loc=None):

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.cv.CV_TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print 'Working on frame %d' % frame_num


def part_1b():
    print "Part 1b"

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = None  # Process noise array
    R = None  # Measurement noise array

    kf = ps5.KalmanFilter(template_loc['x'], template_loc['y'])

    save_frames = {10: os.path.join(output_dir, 'ps5-1-b-1.png'),
                   30: os.path.join(output_dir, 'ps5-1-b-2.png'),
                   59: os.path.join(output_dir, 'ps5-1-b-3.png'),
                   99: os.path.join(output_dir, 'ps5-1-b-4.png')}

    run_kalman_filter(kf,
                      os.path.join(input_dir, "circle"),
                      NOISE_2,
                      "matching",
                      save_frames,
                      template_loc)


def part_1c():
    print "Part 1c"

    init_pos = {'x': 311, 'y': 217}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = None  # Process noise array
    R = None  # Measurement noise array

    kf = ps5.KalmanFilter(init_pos['x'], init_pos['y'])

    save_frames = {10: os.path.join(output_dir, 'ps5-1-c-1.png'),
                   33: os.path.join(output_dir, 'ps5-1-c-2.png'),
                   84: os.path.join(output_dir, 'ps5-1-c-3.png'),
                   159: os.path.join(output_dir, 'ps5-1-c-4.png')}

    run_kalman_filter(kf,
                      os.path.join(input_dir, "walking"),
                      NOISE_1,
                      "hog",
                      save_frames)


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {10: os.path.join(output_dir, 'ps5-2-a-1.png'),
                   33: os.path.join(output_dir, 'ps5-2-a-2.png'),
                   84: os.path.join(output_dir, 'ps5-2-a-3.png'),
                   99: os.path.join(output_dir, 'ps5-2-a-4.png')}

    num_particles = 1000  # Define the number of particles
    sigma_mse = 2  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0  # Set a value for alpha
    dism = 2 # the size of dimensions

    run_particle_filter(ps5.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "circle"),
                        template_loc,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_loc, dism=dism)  # Add more if you need to


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {10: os.path.join(output_dir, 'ps5-2-b-1.png'),
                   33: os.path.join(output_dir, 'ps5-2-b-2.png'),
                   84: os.path.join(output_dir, 'ps5-2-b-3.png'),
                   99: os.path.join(output_dir, 'ps5-2-b-4.png')}

    num_particles = 1000  # Define the number of particles
    sigma_mse = 2  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0  # Set a value for alpha
    dism = 2 # the size of dimensions

    run_particle_filter(ps5.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate_noisy"),
                        template_loc,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_loc, dism=dism)  # Add more if you need to


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {22: os.path.join(output_dir, 'ps5-3-a-1.png'),
                   50: os.path.join(output_dir, 'ps5-3-a-2.png'),
                   160: os.path.join(output_dir, 'ps5-3-a-3.png')}

    num_particles = 1000  # Define the number of particles
    sigma_mse = 2  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.2  # Set a value for alpha
    dism = 2 # the size of dimensions
    run_particle_filter(ps5.AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "pres_debate"),
                        # input video
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect, dism=dism)  # Add more if you need to


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {40: os.path.join(output_dir, 'ps5-4-a-1.png'),
                   100: os.path.join(output_dir, 'ps5-4-a-2.png'),
                   240: os.path.join(output_dir, 'ps5-4-a-3.png'),
                   300: os.path.join(output_dir, 'ps5-4-a-4.png')}

    num_particles = 100  # Define the number of particles
    sigma_md = 2  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 3  # Define the value of sigma for the particles movement (dynamics)
    sigma_scale = 0.015 #Define the value of sigma for the scale factor

    alpha = 0.05
    dism = 3
    run_particle_filter(ps5.MDParticleFilter,
                        os.path.join(input_dir, "pedestrians"),
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_md,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_rect, alpha=alpha,
                        dism=dism, sigma_scale = sigma_scale)  # Add more if you need to


# Frames to record: 29, 56, 71


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    imgs_dir = os.path.join(input_dir, "TUD-Campus")
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    save_frames = {29: os.path.join(output_dir, 'ps5-5-a-1.png'),
                   56: os.path.join(output_dir, 'ps5-5-a-2.png'),
                   71: os.path.join(output_dir, 'ps5-5-a-3.png')}

    # template_rect1 = {'x': 113, 'y': 155, 'w': 30, 'h': 20}
    template_rect1 = {'x': 60, 'y': 215, 'w': 70, 'h': 100}

    # template_rect2 = {'x': 301, 'y': 204, 'w': 20, 'h': 15}
    template_rect2 = {'x': 301, 'y': 248, 'w': 30, 'h': 50}

    template_rect3 = {'x': 205, 'y': 333, 'w': 25, 'h': 70}



    frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
    template_rect_list = {'1':template_rect1, '2':template_rect2, '3':template_rect3}

    template_list = {}
    for key, value in template_rect_list.iteritems():
        template_list[key] = frame[template_rect_list[key]['y']:
                         template_rect_list[key]['y'] + template_rect_list[key]['h'],
                         template_rect_list[key]['x']:
                         template_rect_list[key]['x'] + template_rect_list[key]['w']]

    frame_limit_list = {'1':[1,59], '2':[1,46], '3':[52,71]}

    # Initialize the Algorithm, either KF or PF
    algorithm = 'KF'
    NOISE_1 = {'x': 7.5, 'y': 7.5}
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    noise = NOISE_1
    sensor = "matching"

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        pass


    # Initialize objects
    frame_num = 1
    num_particles = 100  # Define the number of particles
    sigma_mse = 2  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 2  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.2  # Set a value for alpha
    sigma_scale = 0.1 #Define the value of sigma for the scale factor
    dism = 2


    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))


        if algorithm == 'KF':


            for key, value in template_list.iteritems():
                 # Sensor
                if sensor == "hog":
                    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                            padding=(8, 8), scale=1.05)

                    if len(weights) > 0:
                        max_w_id = np.argmax(weights)
                        z_x, z_y, z_w, z_h = rects[max_w_id]

                        z_x += z_w // 2
                        z_y += z_h // 2

                        z_x += np.random.normal(0, noise['x'])
                        z_y += np.random.normal(0, noise['y'])

                elif sensor == "matching":
                    corr_map = cv2.matchTemplate(frame, template_list[key], cv2.cv.CV_TM_SQDIFF)
                    z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

                    z_w = template_rect_list[key]['w']
                    z_h = template_rect_list[key]['h']

                    z_x += z_w // 2 + np.random.normal(0, noise['x'])
                    z_y += z_h // 2 + np.random.normal(0, noise['y'])

                maxframe = frame_limit_list[key][-1]
                minframe = frame_limit_list[key][0]
                frame_num = frame_num

                if (frame_num <= maxframe) & (frame_num >= minframe):

                    kf = ps5.KalmanFilter(template_rect_list[key]['x'], template_rect_list[key]['y'])
                    x, y = kf.process(z_x, z_y)
                    if True:  # For debugging, it displays every frame

                        cv2.circle(frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
                        cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 2)
                        cv2.rectangle(frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                                      (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                                      (0, 0, 255), 2)

                        cv2.imshow('Tracking', frame)
                        cv2.waitKey(1)
                        # cv2.imwrite(save_frames[frame_num], frame)

            # Update frame number
            frame_num += 1
            if frame_num % 20 == 0:
                print 'Working on frame %d' % frame_num

            if frame_num in save_frames:
                cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 2)
                # pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame)


        if algorithm == 'PF':

            for key, value in template_list.iteritems():

                # Extract template and initialize (one-time only)

                template = frame[int(template_rect_list[key]['y']):
                                 int(template_rect_list[key]['y'] + template_rect_list[key]['h']),
                                 int(template_rect_list[key]['x']):
                                 int(template_rect_list[key]['x'] + template_rect_list[key]['w'])]

                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)

                maxframe = frame_limit_list[key][-1]
                minframe = frame_limit_list[key][0]
                frame_num = frame_num

                if (frame_num <= maxframe) & (frame_num >= minframe):
                    pf = ps5.AppearanceModelPF(frame, template,  num_particles=num_particles, sigma_exp=sigma_mse,
                                            sigma_dyn=sigma_dyn, alpha=alpha,
                                            template_coords=template_rect_list[key],dism=dism)
                    # pf = ps5.MDParticleFilter(frame, template,  num_particles=num_particles, sigma_exp=sigma_mse,
                    #                         sigma_dyn=sigma_dyn, alpha=alpha,
                    #                         template_coords=template_rect_list[key],dism=dism,sigma_scale = sigma_scale)
                    # Process frame
                    pf.process(frame)

                    if True:  # For debugging, it displays every frame

                        pf.render(frame)
                        cv2.imshow('Tracking', frame)
                        cv2.waitKey(1)
                else:
                    continue

                        # Render and save output, if indicated
            if frame_num in save_frames:
                # pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame)

            frame_num += 1
            if frame_num % 20 == 0:
                print 'Working on frame %d' % frame_num





    # raise NotImplementedError


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    imgs_dir = os.path.join(input_dir, "follow")
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    save_frames = {60: os.path.join(output_dir, 'ps5-6-a-1.png'),
                   160: os.path.join(output_dir, 'ps5-6-a-2.png'),
                   186: os.path.join(output_dir, 'ps5-6-a-3.png')}

    # template_rect1 = {'x': 113, 'y': 155, 'w': 30, 'h': 20}
    template_rect = {'x': 91, 'y': 88, 'w': 35, 'h': 30}

    frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))


    template = frame[int(template_rect['y']):
                     int(template_rect['y'] + template_rect['h']),
                     int(template_rect['x']):
                     int(template_rect['x'] + template_rect['w'])]



    # Initialize the Algorithm, either KF or PF
    algorithm = 'KF'
    NOISE_1 = {'x': 7.5, 'y': 7.5}
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    noise = NOISE_1
    sensor = "matching"

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        pass


    # Initialize objects
    frame_num = 1
    num_particles = 1000  # Define the number of particles
    sigma_mse = 5  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.2  # Set a value for alpha
    sigma_scale = 0.1 #Define the value of sigma for the scale factor
    dism = 3

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))


        if frame_num < 80:
            algorithm == 'KF'

            # Sensor
            if sensor == "hog":
                (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                        padding=(8, 8), scale=1.05)

                if len(weights) > 0:
                    max_w_id = np.argmax(weights)
                    z_x, z_y, z_w, z_h = rects[max_w_id]

                    z_x += z_w // 2
                    z_y += z_h // 2

                    z_x += np.random.normal(0, noise['x'])
                    z_y += np.random.normal(0, noise['y'])

            elif sensor == "matching":
                corr_map = cv2.matchTemplate(frame, template, cv2.cv.CV_TM_SQDIFF)
                z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

                z_w = template_rect['w']
                z_h = template_rect['h']

                z_x += z_w // 2 + np.random.normal(0, noise['x'])
                z_y += z_h // 2 + np.random.normal(0, noise['y'])



            kf = ps5.KalmanFilter(template_rect['x'], template_rect['y'])
            x, y = kf.process(z_x, z_y)
            if True:  # For debugging, it displays every frame

                cv2.circle(frame, (int(z_x), int(z_y)), 20, (0, 255, 255), 2)
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                              (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                              (0, 255, 255), 2)

                cv2.imshow('Tracking', frame)
                cv2.waitKey(1)
                            # cv2.imwrite(save_frames[frame_num], frame)

                # Update frame number
                frame_num += 1
                if frame_num % 20 == 0:
                    print 'Working on frame %d' % frame_num

                if frame_num in save_frames:
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), 2)
                    # pf.render(frame_out)
                    cv2.imwrite(save_frames[frame_num], frame)

        else:
            algorithm == 'PF'



            # Extract template and initialize (one-time only)


            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            # pf = ps5.AppearanceModelPF(frame, template,  num_particles=num_particles, sigma_exp=sigma_mse,
            #                         sigma_dyn=sigma_dyn, alpha=alpha,
            #                         template_coords=template_rect,dism=dism)
            pf = ps5.MDParticleFilter(frame, template,  num_particles=num_particles, sigma_exp=sigma_mse,
                                    sigma_dyn=sigma_dyn, alpha=alpha,
                                    template_coords=template_rect,dism=dism,sigma_scale = sigma_scale)
            # Process frame
            pf.process(frame)

            if True:  # For debugging, it displays every frame

                pf.render(frame)
                cv2.imshow('Tracking', frame)
                cv2.waitKey(1)


                        # Render and save output, if indicated
            if frame_num in save_frames:
                # pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame)

            frame_num += 1
            if frame_num % 20 == 0:
                print 'Working on frame %d' % frame_num



    # raise NotImplementedError

if __name__ == '__main__':
    part_1b()
    part_1c()
    part_2a()
    part_2b()
    part_3()
    part_4()
    part_5()
    part_6()