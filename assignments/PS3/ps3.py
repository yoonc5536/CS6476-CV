"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import os
# import sys
# sys.path.append('C:\Users\zz\Dropbox (Partners HealthCare)\CS6476CV\assignments')
# from assignments.utility import *
# from utility import *
# print cv2.__version__

def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    a = np.array(p0)
    b = np.array(p1)
    dist = np.linalg.norm(a-b)

    return dist

    # raise NotImplementedError


def corners_detection_distance(res, threshold1, threshold2, image):
    # cv2.imshow('blurred', res)
    # cv2.waitKey(0)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #
    # max_thresh = (max_val + 1e-6) * 0.8
    # loc = np.where(res>=max_thresh)
    max=np.max(res)
    loc = np.where( res >= threshold1*max)
    # yy=loc[::-1]
    locations = zip(*loc[::-1])
    # locations = np.asarray(locations)
    # locations = np.float32(locations)
    # locations=np.matrix([[1062, 466],[979, 101],[197, 290],[283, 638]])
    # locations=np.array[[1062 466][979 101][197 290][283 638]]
    all_potential=dict()
    for iterations in range(0,50):
        if len(locations)>1:

            ran_num = np.random.randint(0, len(locations)-1, size=1)
            # locations = np.random.shuffle(locations)
            exemplar=locations.pop(ran_num)
            # locations.remove(exemplar)
            # locals()['temp_{0}'.format(x)] = []
            temp = [location for location in locations if euclidean_distance(location, exemplar)<threshold2]
                    # locals()['temp_{0}'.format(x)].append(location)
            all_potential[exemplar] = temp
            locations = [x for x in locations if x not in temp]
            # locations.remove(temp)
        else:
            continue

    first_four = [k for k in sorted(all_potential, key=lambda k: len(all_potential[k]), reverse=True)][:4]

    possible_corners=[]
    for key in first_four:
        xxx = np.mean(zip(*all_potential[key])[0])
        yyy = np.mean(zip(*all_potential[key])[1])
        possible_corners.append((xxx,yyy))

    # print all_potential
    # print possible_corners
    return possible_corners

def corners_detection_kmeans(res, threshold1, threshold2, image, verbose):
    max=np.max(res)
    std = np.std(res)
    mean=np.mean(res)
    loc = np.where( res >= threshold1*max)
    locations = zip(*loc[::-1])

    #remove points close to corners
    corners=get_corners_list(image)

    for corner in corners:
        locations=[location for location in locations if euclidean_distance(location, corner)>10]


    locations = np.asarray(locations)
    locations = np.float32(locations)
    # yy=loc[::-1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    # criteria = (cv2.TERM_CRITERIA_EPS, 0, 1)

    ret,label,centers=cv2.kmeans(locations,4,criteria,100,cv2.KMEANS_PP_CENTERS)
    A = locations[label.ravel()==0]
    B = locations[label.ravel()==1]
    C = locations[label.ravel()==2]
    D = locations[label.ravel()==3]
    A = np.uint(A)
    B = np.uint(B)
    C = np.uint(C)
    D = np.uint(D)
    centers=np.uint(centers)

    if verbose:
        cv2.circle(image,(centers[0][0],centers[0][1]),2,(255,0,0),2)
        cv2.circle(image,(centers[1][0],centers[1][1]),2,(0,255,0),2)
        cv2.circle(image,(centers[2][0],centers[2][1]),2,(0,255,255),2)
        cv2.circle(image,(centers[3][0],centers[3][1]),2,(0,255,0),2)
        for i in A:
            image[i[1]][i[0]] = [255,0,0]
        for i in B:
            image[i[1]][i[0]] = [0,255,0]

        for i in C:
            image[i[1]][i[0]] = [0,255,255]

        for i in D:
            image[i[1]][i[0]] = [111,111,111]

        cv2.imshow('res by groups',image)
        cv2.waitKey(0)
    possible_corners=[]
    for center in centers:
        possible_corners.append((center[0],center[1]))
    return possible_corners

def corners_detection_usecircle(res, threshold1, threshold2, image, circle_centers):
    # cv2.imshow('blurred', res)
    # cv2.waitKey(0)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #
    # max_thresh = (max_val + 1e-6) * 0.8
    # loc = np.where(res>=max_thresh)
    max=np.max(res)
    loc = np.where( res >= threshold1*max)
    # yy=loc[::-1]
    locations = zip(*loc[::-1])
    all_potential=dict()
    for x in range(len(circle_centers)):
        # locals()['temp_{0}'.format(x)] = []
        temp=[]
        for location in locations:
            if euclidean_distance(circle_centers[x], location)<15:
                # locals()['temp_{0}'.format(x)].append(location)
                temp.append(location)
            else:
                continue
        all_potential[x] = temp

    possible_corners=[]
    for key in all_potential.keys():
        xxx = np.mean(zip(*all_potential[key])[0])
        yyy = np.mean(zip(*all_potential[key])[1])
        possible_corners.append((xxx,yyy))

    return possible_corners


def corners_detection_template(res, image):
    # indices =  np.argpartition(res.flatten(), -4)[-4:]
    # loc2=np.vstack(np.unravel_index(indices, res.shape)).T
    loc3=np.dstack(np.unravel_index(np.argsort(res.ravel()), res.shape))
    loc4=np.squeeze(loc3)
    loc5 = loc4[::-1]
    possible_corners=[]
    for i in range(len(loc5)):
        if i==0:
            possible_corners.append(loc5[i])
        else:
            if len(possible_corners)==4:
                break
            else:
                if all(np.sqrt((loc5[i][0] - e[0])**2 + (loc5[i][1] - e[1])**2 )>20 for e in possible_corners):
                    possible_corners.append(loc5[i])
                else:
                    continue
    return possible_corners



def circles_check(image, circles, verbose):
    coordinates=[]
    circle_centers=[]
    if circles is not None:
        circles=circles.astype(int)
        for i in circles[0,:]:
            x=i[0]
            y=i[1]
            r=i[2]
            coordinates.append((x,y,r))
            circle_centers.append((x,y))
            coordinates = sorted(coordinates, key=lambda x: x[0])
    if verbose:
        for i in coordinates:
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),i[2],(255,0,255),2)
            # draw the center of the circle
            cv2.circle(image,(i[0],i[1]),2,(255,0,255),3)
        cv2.imshow('Detected Circles', image)
        cv2.waitKey(0)
    return circle_centers, coordinates


def corners_to_markers(method, possible_corners, image, template):
    markers=[]

    for i in range(len(possible_corners)):
        # if method=='matchTemplate' and template is not None:
        #     x = possible_corners[i][1]+template.shape[1]/2
        #     y = possible_corners[i][0]+template.shape[0]/2
        # elif method=='goodFeaturesToTrack':
        #     x = possible_corners[i][0]
        #     y = possible_corners[i][1]
        # else:
        x = possible_corners[i][0]
        y = possible_corners[i][1]
        markers.append((x,y))

    H,W=image.shape[0],image.shape[1]

    ## sort by x value , return index
    x_index = sorted(range(len(markers)), key=lambda x: markers[x][0])
    y_index = sorted(range(len(markers)), key=lambda x: markers[x][1])
    left = x_index[:2]
    right = x_index[-2:]
    top = y_index[:2]
    bottom = y_index[-2:]
    if sorted(left) == sorted(top):
        if H <= W:
            top_left_index=left[0]
            top_right_index=left[1]
            bottom_right_index=right[1]
            bottom_left_index=right[0]
        if H > W:
            top_left_index=top[0]
            top_right_index=top[1]
            bottom_right_index=bottom[1]
            bottom_left_index=bottom[0]

    elif sorted(left) == sorted(bottom):
        if H <= W:
            top_left_index=bottom[0]
            bottom_left_index=bottom[1]
            top_right_index=top[0]
            bottom_right_index=top[1]
        if H > W:
            top_left_index=right[1]
            bottom_left_index=right[0]
            top_right_index=left[1]
            bottom_right_index=left[0]

    else:
        top_left_index = list(set(top) - (set(top) - set(left)))[0]
        top_right_index = list(set(top) - (set(top) - set(right)))[0]
        bottom_left_index = list(set(bottom) - (set(bottom) - set(left)))[0]
        bottom_right_index = list(set(bottom) - (set(bottom) - set(right)))[0]

    top_left = markers[top_left_index]
    top_right = markers[top_right_index]
    bottom_left = markers[bottom_left_index]
    bottom_right = markers[bottom_right_index]





    # if H <= W:

    # corners=get_corners_list(image)
    #
    # for corner in corners:
    #     locations=[location for location in locations if euclidean_distance(location, corner)>10]

    #     distance=[i[0]**2 + i[1]**2 for i in markers]
    #     index_min = np.argmin(distance)
    #     index_max=np.argmax(distance)
    #     top_left=markers[index_min]
    #     bottom_right=markers[index_max]
    #     rest = [x for x in markers if x != top_left and x!=bottom_right]
    #     rest.sort(key=lambda x: x[0])
    #     bottom_left=rest[0]
    #     top_right=rest[1]
    # else:
    #     distance=[(i[0]-W)**2 + i[1]**2 for i in markers]
    #     index_min = np.argmin(distance)
    #     index_max=np.argmax(distance)
    #     top_left=markers[index_min]
    #     bottom_right=markers[index_max]
    #     rest = [x for x in markers if x != top_left and x!=bottom_right]
    #     rest.sort(key=lambda x: x[1])
    #     bottom_left=rest[0]
    #     top_right=rest[1]

    # top_left=
    # bottom_left=(max_loc[0],max_loc[1]+H)
    # bottom_right=min_loc
    # top_right=(bottom_right[0],bottom_right[1]-H)
    final_markers=[top_left, bottom_left, top_right, bottom_right]
    return final_markers

# reference: https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
def bilinear_interpolate(im, x, y):
    # x = np.asarray(x)
    # y = np.asarray(y)

    # shape = im.shape
    x = np.clip(x, 0, im.shape[1]-1)
    y = np.clip(y, 0, im.shape[0]-1)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-2)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-2)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id




def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    x0=0
    y0=0
    top_left=(x0,y0)
    h,w=image.shape[0],image.shape[1]
    bottom_left=(x0,y0+h-1)
    top_right=(x0+w-1,y0)
    bottom_right=(x0+w-1,y0+h-1)
    corners=[top_left,bottom_left,top_right,bottom_right]
    return corners
    # raise NotImplementedError


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """


    verbose = False

    # method:  'matchTemplate' 'cornerMinEigenVal' 'cornerHarris'
    # algos: 'kmeans','distance','template'
    # cornerHarris, kmeans
    # 0.2
    # 5
    # 10, 3, 0.001

    method = 'cornerHarris'
    algos = 'kmeans'
    if template is not None:
        w = template.shape[1]/2
        h = template.shape[0]/2
    threshold1=0.2
    threshold2=50




    H,W=image.shape[0],image.shape[1]
    result=image.copy()

    # kernel = np.ones((5,5),np.float32)/25
    # image = cv2.filter2D(image,-1,kernel)
    image = cv2.GaussianBlur(image,(5,5),0)
    # blurred = cv2.blur(image,(5,5))
    blurred = cv2.medianBlur(image,5)
    # blurred = cv2.GaussianBlur(image, (5,5), 0)

    # image = cv2.bilateralFilter(image,9,75,75)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # Part2 80, 200, (300 60, 25)
    # Test 60, 120, 150, 50, 25
    # edges = cv2.Canny(gray,60,150)
    # cv2.imshow('mask', mask)
    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    # template_edges=cv2.Canny(template,100,200)

    # circles = cv2.HoughCircles(edges,cv2.cv.CV_HOUGH_GRADIENT,1,150,param1=50,param2=25,minRadius=10,maxRadius=100)
    # circle_centers,coordinates=circles_check(image, circles, verbose)



    # corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=4, qualityLevel=0.01, minDistance=100, blockSize=3, useHarrisDetector=0, k=0.04)
    # possible_corners=np.squeeze(corners)


    if method == 'matchTemplate':
        res = cv2.matchTemplate(blurred,template,eval('cv2.TM_CCOEFF_NORMED'))
    elif method == 'cornerHarris':
        res = cv2.cornerHarris(gray, 10, 3, 0.001)
        res = cv2.normalize(res,        # src
                            None,             # dst
                            0,                # alpha
                            255,              # beta
                            cv2.NORM_MINMAX,  # norm type
                            cv2.CV_32FC1,     # dtype
                            None              # mask
                            )
        # res = cv2.dilate(res,(5,5))
    elif method == 'cornerMinEigenVal':
        # res=cv2.preCornerDetect(gray, 3)
        res = cv2.cornerMinEigenVal(gray,10,5)

    if algos == 'template':
        possible_corners=corners_detection_template(res,image)
        # final_markers=corners_to_markers('matchTemplate',possible_corners,template)
    elif algos == 'kmeans':
        possible_corners=corners_detection_kmeans(res,threshold1,threshold2,image,verbose=verbose)
    elif algos == 'distance':
        possible_corners=corners_detection_distance(res, threshold1, threshold2, image)
    final_markers=corners_to_markers(method,possible_corners,image,template)

    # if circles is not None and len(circle_centers) == 4:
    #     res = cv2.cornerMinEigenVal(gray,10,5)
    #     possible_corners=corners_detection_engi(res, threshold1,threshold2, image, circle_centers)
    #     final_markers=corners_to_markers('cornerMinEigenVal',possible_corners,w,h)
    # else:
    #     res = cv2.matchTemplate(image,template,eval('cv2.TM_CCOEFF_NORMED'))
    #     possible_corners=corners_detection_template(res,image)
    #     final_markers=corners_to_markers('matchTemplate',possible_corners,w,h)

    final_markers_int = [tuple(int(x) for x in tup) for tup in final_markers]


    if verbose:
        image[res>threshold1*res.max()]=[0,0,255]
        cv2.imshow('dst',image)
        color = (0, 50, 255)
        for pt in final_markers_int:
            cv2.circle(result, pt, 3, color, -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)
        cv2.imshow('final', result)
        cv2.waitKey(0)


    return final_markers_int



    # raise NotImplementedError


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    a = markers[0::2]
    b = markers[1::2]
    x = [tuple(int(x) for x in tup) for tup in a]
    y = [tuple(int(x) for x in tup) for tup in b]

    pairs= zip(x,y)
    pairs1=zip(*pairs)
    pairs2=pairs+pairs1
    for points in pairs2:
        cv2.line(image, points[0], points[1], (255), thickness=thickness)
    # reorder_markers=[]
    # reorder_markers.append(markers[0:2])
    # reorder_markers.append(markers[-1])
    # reorder_markers.append(markers[2])
    # points = np.asarray(reorder_markers)
    # cv2.polylines(image, np.int32([points]), thickness, (255))
    # cv2.imshow('imag', image)
    # cv2.waitKey(0)
    return image
    # raise NotImplementedError


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    verbose = False
    # template = cv2.imread("input_images/template.jpg")

    x0=0
    y0=0
    h,w=imageB.shape[0],imageB.shape[1]



    H = np.linalg.inv(homography)
    combined=imageB.copy()

    # xs = zip(*markers)[0]
    # ys = zip(*markers)[1]

    for i in range(x0, x0+w):
        for j in range(y0, y0+h):
            # x_prime, y_prime=markers[j][0], markers[j][1]
            # x_prime, y_prime=markers[0][0]+i, markers[0][1]+j
            x_prime, y_prime=i, j

            # The target point
            p_prime = np.array([[x_prime],[y_prime],[1]])
            # The source point based on inverse homography
            p = np.dot(H, p_prime)
            p = p/p[-1]
            if ((p[0] >=0) & (p[0] <= imageA.shape[1])) & ((p[1] >= 0) & (p[1] <= imageA.shape[0])):
                chk = bilinear_interpolate(imageA,p[0], p[1])
                combined[y_prime,x_prime] = chk
            else:
                continue

    if verbose:
    #     for i in markers:
    #         cv2.circle(imageB,(i[0],i[1]),2,(255,0,255),3)
    #     cv2.imshow('Detected Corners', imageB)
    #     cv2.waitKey(0)
        cv2.imshow('combined',combined)
        cv2.waitKey(0)
    return combined



    # raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    # min Ah-b
    A = []
    b= []
    for i in range(len(src_points)):
        x, y = src_points[i][0], src_points[i][1]
        u, v = dst_points[i][0], dst_points[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        b.append([u])
        b.append([v])
    A = np.asarray(A)
    b = np.asarray(b)
    # transform_matrix = np.linalg.solve(A,b)
    H = np.linalg.lstsq(A,b)[0]
    h9 = np.array([1])
    H = np.append(H,h9)
    H = H.reshape(3,3)
    # U, S, Vh = np.linalg.svd(A)
    # L = Vh[-1,:] / Vh[-1,-1]
    # H = L.reshape(3, 3)
    return H

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
    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    # raise NotImplementedError



# IMG_DIR = "input_images"
# VID_DIR = "input_videos"
# OUT_DIR = "output"
# if not os.path.isdir(OUT_DIR):
#     os.makedirs(OUT_DIR)
#
# def helper_for_part_4_and_5(video_name, fps, frame_ids, output_prefix,
#                             counter_init, is_part5):
#
#     video = os.path.join(VID_DIR, video_name)
#     image_gen = video_frame_generator(video)
#
#     image = image_gen.next()
#     h, w, d = image.shape
#
#     out_path = "output/ar_{}-{}".format(output_prefix[4:], video_name)
#     video_out = mp4_video_writer(out_path, (w, h), fps)
#
#     # Optional template image
#     template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
#
#     if is_part5:
#         advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
#         src_points = get_corners_list(advert)
#
#     output_counter = counter_init
#
#     frame_num = 1
#
#     while image is not None:
#
#         print "Processing fame {}".format(frame_num)
#
#         markers = find_markers(image, template)
#
#         if is_part5:
#             homography = find_four_point_transform(src_points, markers)
#             image = project_imageA_onto_imageB(advert, image, homography)
#
#         else:
#
#             for marker in markers:
#                 mark_location(image, marker)
#
#         frame_id = frame_ids[(output_counter - 1) % 3]
#
#         if frame_num == frame_id:
#             out_str = output_prefix + "-{}.png".format(output_counter)
#             save_image(out_str, image)
#             output_counter += 1
#
#         video_out.write(image)
#
#         image = image_gen.next()
#
#         frame_num += 1
#
#     video_out.release()
#
#
# def mp4_video_writer(filename, frame_size, fps=20):
#     """Opens and returns a video for writing.
#
#     Use the VideoWriter's `write` method to save images.
#     Remember to 'release' when finished.
#
#     Args:
#         filename (string): Filename for saved video
#         frame_size (tuple): Width, height tuple of output video
#         fps (int): Frames per second
#     Returns:
#         VideoWriter: Instance of VideoWriter ready for writing
#     """
#     # fourcc = cv2.cv.CV_FOURCC(*'MP4V')
#     fourcc = cv2.cv.CV_FOURCC(*'MJPG')
#     filename = filename.replace('.mp4', '.avi')
#     return cv2.VideoWriter(filename, fourcc, fps, frame_size)
#
#
# def save_image(filename, image):
#     """Convenient wrapper for writing images to the output directory."""
#     cv2.imwrite(os.path.join(OUT_DIR, filename), image)
#
#
# def mark_location(image, pt):
#     """Draws a dot on the marker center and writes the location as text nearby.
#
#     Args:
#         image (numpy.array): Image to draw on
#         pt (tuple): (x, y) coordinate of marker center
#     """
#     color = (0, 50, 255)
#     cv2.circle(image, pt, 3, color, -1)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)
#
# def part_3():
#
#     print "\nPart 3:"
#
#     input_images = ['ps3-3-a_base.jpg', 'ps3-3-b_base.jpg', 'ps3-3-c_base.jpg']
#     output_images = ['ps3-3-a-1.png', 'ps3-3-a-2.png', 'ps3-3-a-3.png']
#
#     # input_images = ['ps3-3-c_base.jpg']
#     # output_images = ['ps3-3-a-3.png']
#
#
#     # Advertisement image
#     advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
#     src_points = get_corners_list(advert)
#
#     # Optional template image
#     template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
#
#     for img_in, img_out in zip(input_images, output_images):
#         print "Input image: {}".format(img_in)
#
#         # Open image and identify the four marker positions
#         scene = cv2.imread(os.path.join(IMG_DIR, img_in))
#
#         markers = find_markers(scene, template)
#
#         homography = find_four_point_transform(src_points, markers)
#
#         projected_img = project_imageA_onto_imageB(advert, scene,
#                                                        homography)
#
#         save_image(img_out, projected_img)
#
# def part_4_a():
#
#     print "\nPart 4a:"
#
#     video_file = "ps3-4-a.mp4"
#     frame_ids = [355, 555, 725]
#     fps = 40
#
#     # helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 1, False)
#
#     video_file = "ps3-4-b.mp4"
#     frame_ids = [97, 407, 435]
#     fps = 40
#
#     helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 4, False)
#
# def part_4_b():
#
#     print "\nPart 4b:"
#
#     video_file = "ps3-4-c.mp4"
#     frame_ids = [47, 470, 691]
#     fps = 40
#
#     helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-b", 1, False)
#
#     video_file = "ps3-4-d.mp4"
#     frame_ids = [207, 367, 737]
#     fps = 40
#
#     helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-b", 4, False)
#
# def part_5_a():
#
#     print "\nPart 5a:"
#
#     video_file = "ps3-4-a.mp4"
#     frame_ids = [355, 555, 725]
#     fps = 40
#
#     helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-a", 1, True)
#
#     video_file = "ps3-4-b.mp4"
#     frame_ids = [97, 407, 435]
#     fps = 40
#
#     helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-a", 4, True)
#
# def part_5_b():
#
#     print "\nPart 5b:"
#
#     video_file = "ps3-4-c.mp4"
#     frame_ids = [47, 470, 691]
#     fps = 40
#
#     helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-b", 1, True)
#
#     video_file = "ps3-4-d.mp4"
#     frame_ids = [207, 367, 737]
#     fps = 40
#
#     # helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-b", 4, True)
#
#
# # video_file, fps, frame_ids, "ps3-5-b", 1
# def helper_for_part_6(video_name, my_video, fps, frame_ids, output_prefix,
#                             counter_init):
#
#     video = os.path.join(VID_DIR, video_name)
#     my_video = os.path.join(VID_DIR, my_video)
#     image_gen = video_frame_generator(video)
#     my_video_image_gen = video_frame_generator(my_video)
#
#     image = image_gen.next()
#     h, w, d = image.shape
#     my_video_image = my_video_image_gen.next()
#
#     out_path = "output/ar_{}-{}".format(output_prefix[4:], video_name)
#     video_out = mp4_video_writer(out_path, (w, h), fps)
#
#     # Optional template image
#     template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
#
#
#
#     output_counter = counter_init
#
#     frame_num = 1
#
#     src_points = get_corners_list(my_video_image)
#     while image is not None:
#
#         print "Processing fame {}".format(frame_num)
#
#         markers = find_markers(image, template)
#
#         # advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
#         # src_points = get_corners_list(advert)
#         homography = find_four_point_transform(src_points, markers)
#         image = project_imageA_onto_imageB(my_video_image, image, homography)
#
#
#         frame_id = frame_ids[(output_counter - 1) % 3]
#
#         if frame_num == frame_id:
#             out_str = output_prefix + "-{}.png".format(output_counter)
#             save_image(out_str, image)
#             output_counter += 1
#
#         video_out.write(image)
#
#         image = image_gen.next()
#         my_video_image = my_video_image_gen.next()
#
#         frame_num += 1
#
#     video_out.release()
#
#
# def part_6():
#
#     print "\nPart 6:"
#
#     video_file = "ps3-4-a.mp4"
#     my_video = "my-ad.mp4"  # Place your video in the input_video directory
#     frame_ids = [355, 555, 725]
#     fps = 40
#
#     helper_for_part_6(video_file, my_video, fps, frame_ids, "ps3-6", 1)
#
# if __name__ == '__main__':

    # part_3()
    # part_4_a()
    # part_4_b()
    # part_5_b()
    # part_6()
    # scene = cv2.imread(os.path.join(IMG_DIR, "sim_clear_scene - Copy.jpg"))
    # template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
    # markers = find_markers(scene, template)
    # print "markers", markers
    # image_with_box = draw_box(scene, markers, 3)
    # img_out="testtt.png"
    # save_image(img_out, image_with_box)
    # advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
    # scene = cv2.imread(os.path.join(IMG_DIR, 'ps3-3-a_base.jpg'))
    # src_points = [(0,0),(0,480),(852,0),(852,480)]
    # dst_points = [(105,85),(100,676),(944,175),(929,555)]
    # homography = find_four_point_transform(src_points, dst_points)
    # projected_img = project_imageA_onto_imageB(advert, scene,
    #                                                    homography)
    # cv2.imshow("projected",projected_img)
    # cv2.waitKey(0)
    # save_image('ps3-3-a-1.png', projected_img)




