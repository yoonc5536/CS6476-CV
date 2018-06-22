import cv2
import numpy as np


def intersec(line1, line2):
        noise1=np.asscalar(0.00001*np.random.rand(1, 1))
        noise2=np.asscalar(0.00001*np.random.rand(1, 1))

        A = [[np.cos(line1[1])+noise1,np.sin(line1[1])+noise2],[np.cos(line2[1]),np.sin(line2[1])]]
#         print A
        b = [line1[0],line2[0]]

#         print b
        intersection = np.dot(np.linalg.inv(A), b)
        intersection = intersection.astype(int)
        return intersection


def houghline(img, threshold):
    lines = cv2.HoughLines(img,1,np.pi/180, threshold)
    coordinates=[]
    if lines is not None:
        for i in lines[0,:]:
            rho=i[0]
            theta=i[1]
            coordinates.append((rho,theta))
    return coordinates


def lines_check(img_in, coordinates):
    result = img_in.copy()
    #QC Part
    for rho, theta in coordinates:
          # print rho
#         # print theta
#         if (theta - np.pi/3)
        if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #vertical line
            #The intersection with the first row
            pt1 = (int(rho/np.cos(theta)),0)
#             # print "vertical up", pt1
            #The intersection with the last row
            pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
            #draw a white line
#             # print "vertical down", pt2
            cv2.line( result, pt1, pt2, (255))
        else: #horizontal line
            #The intersection with the first column
            pt1 = (0,int(rho/np.sin(theta)))
#             # print "horizontal left", pt1
            #The intersection with the last column
            pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
            #draw a white line
#             # print "horizontal right", pt2
            cv2.line(result, pt1, pt2, (255), 1)




class Render(object):
    def __init__(self, function, constants, variables):
        """Accepts
        @param function: which gets inputs 'constants' and 'variables'
        @param constants: to the function like images
        @param variables: dictionary of variables.
                            The keys are variables and values are max values. min is 0.
                            increments in steps of 1 from min->max
        """
        self.window_name = 'Param Tuning'

        self.function = function
        self.constants = constants
        self.variables = variables

        cv2.namedWindow(self.window_name)
        for varK, varV in self.variables.items():
            cv2.createTrackbar(varK, 'Param Tuning', 0, varV, self.execute)

    def execute(self, val=None):
        """Renders the window with original image and processed image with track bars"""
        # - Get the current positions on track bars
        for var in self.variables.keys():
            self.variables[var]=cv2.getTrackbarPos(var, self.window_name)
        self.variables['window']=self.window_name

        # - Execute function with current track bar values
        # self.function(**self.variables)
        try:
            print self.variables
            self.function(*self.constants, **self.variables)
        except np.linalg.LinAlgError:
            print 'Singular Matrix found...'
        except:
            print 'Some other error... CHECK...'

        cv2.waitKey(0)
        cv2.destroyAllWindows()

