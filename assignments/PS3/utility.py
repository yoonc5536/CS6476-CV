import numpy as np
import cv2


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
    std = np.std(res)
    mean=np.mean(res)
    # threshold = 0.2
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
            temp = []
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

    # possible_corners=[]
    # for center in centers:
    #     possible_corners.append((center[0],center[1]))
    # print all_potential
    # print possible_corners
    return possible_corners

def corners_detection_kmeans(res, threshold1, threshold2, image, circle_centers):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    criteria = (cv2.TERM_CRITERIA_EPS, 0, 1)

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

        cv2.imshow('A',image)
        cv2.waitKey(0)


    # indices =  np.argpartition(res.flatten(), -4)[-4:]
    # loc2=np.vstack(np.unravel_index(indices, res.shape)).T
    # loc2 = res.argsort()[-3:][::-1]

    possible_corners=[]


def corners_detection_engi(res, threshold1, threshold2, image, circle_centers):
    # cv2.imshow('blurred', res)
    # cv2.waitKey(0)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #
    # max_thresh = (max_val + 1e-6) * 0.8
    # loc = np.where(res>=max_thresh)

    loc = np.where( res >= threshold1*max)

    # yy=loc[::-1]
    locations = zip(*loc[::-1])
    # locations = np.asarray(locations)
    # locations = np.float32(locations)

    # locations=np.matrix([[1062, 466],[979, 101],[197, 290],[283, 638]])

    # locations=np.array[[1062 466][979 101][197 290][283 638]]
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

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    # criteria = (cv2.TERM_CRITERIA_EPS, 0, 1)
    #
    # ret,label,centers=cv2.kmeans(locations,4,criteria,100,cv2.KMEANS_PP_CENTERS)
    # # centers = np.int(centers)
    # A = locations[label.ravel()==0]
    # B = locations[label.ravel()==1]
    # C = locations[label.ravel()==2]
    # D = locations[label.ravel()==3]
    # A = np.uint(A)
    # B = np.uint(B)
    # C = np.uint(C)
    # D = np.uint(D)
    # centers=np.uint(centers)
    #
    # cv2.circle(image,(centers[0][0],centers[0][1]),2,(255,0,0),2)
    # cv2.circle(image,(centers[1][0],centers[1][1]),2,(0,255,0),2)
    # cv2.circle(image,(centers[2][0],centers[2][1]),2,(0,255,255),2)
    # cv2.circle(image,(centers[3][0],centers[3][1]),2,(0,255,0),2)
    # cv2.imshow('A',image)
    # cv2.waitKey(0)
    #
    # for i in A:
    #     image[i[1]][i[0]] = [255,0,0]
    #
    #
    # for i in B:
    #     image[i[1]][i[0]] = [0,255,0]
    #
    # for i in C:
    #     image[i[1]][i[0]] = [0,255,255]
    #
    # for i in D:
    #     image[i[1]][i[0]] = [111,111,111]
    #
    #
    # cv2.imshow('A',image)
    # cv2.waitKey(0)



    # possible_corners=[]
    # for center in centers:
    #     possible_corners.append((center[0],center[1]))
    return possible_corners


def corners_detection_template(res, image):
    max=np.max(res)
    std = np.std(res)
    mean=np.mean(res)
    # indices =  np.argpartition(res.flatten(), -4)[-4:]
    # loc2=np.vstack(np.unravel_index(indices, res.shape)).T
    # loc2 = res.argsort()[-3:][::-1]
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


def corners_to_markers(method, possible_corners, w, h):
    markers=[]
    for i in range(len(possible_corners)):
        if method=='matchTemplate':
            x = possible_corners[i][1]+w/2
            y = possible_corners[i][0]+h/2
        elif method=='goodFeaturesToTrack':
            x = possible_corners[i][0]
            y = possible_corners[i][1]
        else:
            x = possible_corners[i][0]
            y = possible_corners[i][1]
        markers.append((x,y))

    distance=[i[0]**2 + i[1]**2 for i in markers]
    index_min = np.argmin(distance)
    index_max=np.argmax(distance)
    top_left=markers[index_min]
    bottom_right=markers[index_max]
    rest = [x for x in markers if x != top_left and x!=bottom_right]
    rest.sort(key=lambda x: x[0])
    bottom_left=rest[0]
    top_right=rest[1]
    # top_left=
    # bottom_left=(max_loc[0],max_loc[1]+H)
    # bottom_right=min_loc
    # top_right=(bottom_right[0],bottom_right[1]-H)
    final_markers=[top_left, bottom_left, top_right, bottom_right]
    return final_markers

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
