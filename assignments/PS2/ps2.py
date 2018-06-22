
"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
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

def generate_mask(img, bgr):
    
    thresh = 0

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
    
    mask = cv2.inRange(img,minBGR,maxBGR)
    return mask

# /120, 240/
#/300,1200 works all
def generate_edges(img):
    edges = cv2.Canny(img,300,1200)
    return edges

def generate_gray(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def generate_blurred(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred

def houghline(img, threshold):
    lines = cv2.HoughLines(img,1,np.pi/180, threshold)
    coordinates=[]
    if lines is not None:
        for i in lines[0,:]:
            rho=i[0]
            theta=i[1]
            coordinates.append((rho,theta))
    return coordinates

# 50 0 50 
def houghcircles(img, threshold, minR, maxR): 
    circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,threshold,param1=20,param2=10,minRadius=minR,maxRadius=maxR) 
    circle_coordinates=[]
    if circles is not None:
        circles=circles.astype(int)
        for i in circles[0,:]:
            x=i[0]
            y=i[1]
            r=i[2]
            circle_coordinates.append((x,y,r))
            circle_coordinates = sorted(circle_coordinates, key=lambda x: x[0])
    return circle_coordinates

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
        
#     # print lines 
#     cv2.imshow('thresh', thresh)
#     cv2.imshow('Result', result)
#     cv2.waitKey(0)
    
    
def circles_check(result, coordinates):
    # QC PART 

    for i in coordinates:
        # draw the outer circle 
        cv2.circle(result,(i[0],i[1]),i[2],(255,0,255),2) 
        # draw the center of the circle 
        cv2.circle(result,(i[0],i[1]),2,(255,0,255),3) 
        cv2.imshow('detected circles',result) 
    cv2.waitKey(0) 
#     cv2.destroyAllWindows()

#     cv2.circle(result, (772,349), 2, (0,0,255), 2)
#     cv2.circle(result, (800,349), 2, (255,0,0), 2)


    
def find_yield_sign(coordinates):
    """
    Part3: Find the valid lines based on the theta difference
    The logic used:
        1. The angle between two lines should approximate 60
        2. The rho of two lines should differ enough
        3. The side lenghs should approximate enough
    """
    # print "total lines", len(coordinates)
    possible_edges = []
    for line1 in coordinates:
        # print "#####start line1", line1
        for line2 in coordinates: 
            if  (line2!=line1) and abs(line1[1]-line2[1]-np.pi/3.0)<0.001:
                # print "@@@@@find line2", line1, line2
                for line3 in coordinates : 
                    if (line3!=line2 and line3!=line1) and abs(abs(line2[1]-line3[1])-np.pi/3.0)<0.001 \
                    and abs(line3[0]-line2[0])>5 and abs(line3[0]-line1[0])>5:
                        # print "$$$$find line3", line1, line2, line3
                        possible_edges.append((line1, line2, line3))
#                         coordinates = [e for e in coordinates if e not in (line1, line2, line3)]        
                    else:
                        continue
            else:
                continue

#     print "possible edges", possible_edges
    
    possible_vertexs=[]
    
    
    for edges in possible_edges:
#         print "edges", edges
        store_vertexs=[]
        for j in range(3):
#             print "process", edges[-1], edges[0]
            store_vertexs.append(intersec(edges[j-1], edges[j]))
#         # print "current store_vertexs", store_vertexs    
        possible_vertexs.append(store_vertexs)
        
#     print "possible_vertexs", possible_vertexs
    
    for vertexs in possible_vertexs:
        # print "possing vertexs", vertexs
        dist_1 = np.sqrt( (vertexs[1][0] - vertexs[0][0])**2 + (vertexs[1][1] - vertexs[0][1])**2 )
        dist_2 = np.sqrt( (vertexs[2][0] - vertexs[1][0])**2 + (vertexs[2][1] - vertexs[1][1])**2 )
        # print "dist_1", dist_1
        # print "dist_2", dist_2
        if abs(dist_1-dist_2)>3:
            possible_vertexs = [e for e in possible_vertexs if not np.array_equal(e,vertexs)]
    
#     print "possible_vertexs after distance check", possible_vertexs
    
    """
    Part4: more logic to select vertexs
    The logic are: 
        1. The BGR value for the center should match
    """
    possible_centers=[]
    valid_vertexs=[]
    
    for vertexs in possible_vertexs:
        # print "current vertexs", vertexs
    
        center_x = (sum([vertex[0] for vertex in vertexs])/len(vertexs)).astype(int)
        center_y = (sum([vertex[1] for vertex in vertexs])/len(vertexs)).astype(int)
        if abs(center_x) < 1000 and abs(center_y) < 1000:
#             A=img_in[center_y][center_x]

#             B=[255,255,255]
#             if np.array_equal(A,B):
#                 #QC Part, draw vertexs                   
            possible_centers.append((center_x, center_y))
            valid_vertexs.append(vertexs)
    #             break 

#             else:
#                 continue
        else:
            continue

    """Part4: return the outputs
    """ 
    
#     print "after loop", possible_centers
    centers=()
    vertexs=[]
    if len(possible_centers) > 0: 
        centers=possible_centers[0]
        vertexs=valid_vertexs[0]
    return centers, vertexs


def find_stop_sign(coordinates, threshold1):
    """
    Part2: Hough Transform to generate the lines 
    """
    
    
    def edges(weight, threshold):
        edges=[]
        for x in coordinates:
            if abs(x[1]-weight*np.pi/4.0) < threshold: 
                edges.append(x)
        edges = sorted(edges, key=lambda x:x[0])
        return edges

#     def edges(weight):
#         edges = sorted(coordinates, key=lambda x:abs(x[1]-weight*np.pi/4.0))[:2]
        
#         edges = sorted(edges, key=lambda x:x[0])
#         return edges
    
    horizontal_edges=edges(2.0, 0.1)
    left_edges=edges(3.0, 0.1)
    vertical_edges=edges(0.0, 0.02)
    right_edges=edges(1.0, 0.02)

    if len(horizontal_edges)>0 and len(left_edges)>0 and len(vertical_edges)>0 and len(right_edges)>0:
        edge2=left_edges[0]
        edge6=left_edges[-1]
        # print "left", left_edges

        edge3=vertical_edges[-1]
        edge7=vertical_edges[0]
        # print "verti", vertical_edges

        edge4=right_edges[-1]
        edge8=right_edges[0]
        # print "right", right_edges

        dist=abs(edge3[0]-edge7[0])

        # print "horz", horizontal_edges
        possible_hori=dict()
        for i in range(len(horizontal_edges)):
            for j in range(len(horizontal_edges)):
                possible_hori[(i,j)]=abs(horizontal_edges[i][0]-horizontal_edges[j][0])

    #     possible_left=dict()
    #     for i in range(len(left_edges)):
    #         for j in range(len(left_edges)):
    #             possible_left[(i,j)]=abs(left_edges[i][0]-left_edges[j][0])

    #     possible_right=dict()
    #     for i in range(len(right_edges)):
    #         for j in range(len(right_edges)):
    #             possible_right[(i,j)]=abs(right_edges[i][0]-right_edges[j][0])

        # print "possible_hori", possible_hori
        # print "dist", dist
        #logic: the distance between opposite sides should be close
        edge1s=[]
        edge5s=[]
        for key, value in possible_hori.items():
            if abs(dist-value)<threshold1:
                key=list(key)
                key.sort()
    #             print "key", key
                edge1s.append(horizontal_edges[key[0]])
                edge5s.append(horizontal_edges[key[1]])

            else:
                continue

        # print "edg1", edge1s
        # print "edge5", edge5s

        inter3 = intersec(edge2, edge3)
        inter4 = intersec(edge3, edge4)
        inter7 = intersec(edge6, edge7)
        inter8 = intersec(edge7, edge8)


        #logic: the side length should be very close
        side_length = np.sqrt( (inter3[0] - inter4[0])**2 + (inter3[1] - inter4[1])**2 )

        # print "side_", side_length
        for i in range(len(edge1s)):

            inter1 = intersec(edge8, edge1s[i])
            inter2 = intersec(edge1s[i], edge2)
            inter5 = intersec(edge4, edge5s[i])
            inter6 = intersec(edge5s[i], edge6)


            possible_side_length1 = np.sqrt( (inter1[0] - inter2[0])**2 + (inter1[1] - inter2[1])**2 )
            possible_side_length2 = np.sqrt( (inter5[0] - inter6[0])**2 + (inter5[1] - inter6[1])**2 )
            # print "possible", possible_side_length1, possible_side_length2
            if abs(side_length - possible_side_length1) < 8 and abs(side_length - possible_side_length2) < 8:
                # print "current", edge1s[i]
                possible_i=i
                edge1=edge1s[possible_i]
                edge5=edge5s[possible_i]
                break
            else:
                continue





        # print "edges", edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8








    #     edge1=horizontal_edges[0]
    #     edge5=horizontal_edges[4]

    #     for key, value in possible_left.items():
    #         if abs(dist-value)<4:
    #             key=list(key)
    #             key.sort()
    # #             print "key", key
    #             edge2=left_edges[key[0]]
    #             edge6=left_edges[key[1]]
    #             break
    #         else:
    #             continue


    #     for key, value in possible_right.items():
    #         if abs(dist-value)<4:
    #             key=list(key)
    #             key.sort()
    # #             print "key", key
    #             edge4=right_edges[key[0]]
    #             edge8=right_edges[key[1]]
    #             break
    #         else:
    #             continue







    #     line1s=horizontal_edges
    #     line5s=horizontal_edges
    #     line2s=left_edges
    #     line6s=left_edges
    #     line3s=vertical_edges
    #     line7s=vertical_edges
    #     line4s=right_edges
    #     line8s=right_edges

    #     possible_edges=[]
    #     for line1 in line1s:
    #         for line2 in line2s:
    #             for line3 in line3s:
    #                 for line4 in line4s:
    #                     for line5 in line5s:
    #                         for line6 in line6s:
    #                             for line7 in line7s:
    #                                 for line8 in line8s:
    #                                     possible_edges.append((line1, line2, line3, line4,line5,line6,line7,line8))


    #     possible_vertexs=[]
    #     for edges in possible_edges:
    # #         # print "current edges", edges
    #         store_vertexs=[]
    #         for j in range(8):
    #             store_vertexs.append(intersec(edges[j-1], edges[j]))
    # #         # print "current store_vertexs", store_vertexs
    #         possible_vertexs.append(store_vertexs)

    #     for vertexs in possible_vertexs:
    #         # print "possing vertexs", vertexs
    #         dist_1 = np.sqrt( (vertexs[1][0] - vertexs[0][0])**2 + (vertexs[1][1] - vertexs[0][1])**2 )
    #         dist_2 = np.sqrt( (vertexs[2][0] - vertexs[1][0])**2 + (vertexs[2][1] - vertexs[1][1])**2 )
    #         dist_3 = np.sqrt( (vertexs[3][0] - vertexs[2][0])**2 + (vertexs[3][1] - vertexs[2][1])**2 )
    #         dist_4 = np.sqrt( (vertexs[4][0] - vertexs[3][0])**2 + (vertexs[4][1] - vertexs[3][1])**2 )
    #         dist_5 = np.sqrt( (vertexs[5][0] - vertexs[4][0])**2 + (vertexs[5][1] - vertexs[4][1])**2 )
    #         dist_6 = np.sqrt( (vertexs[6][0] - vertexs[5][0])**2 + (vertexs[6][1] - vertexs[5][1])**2 )
    #         dist_7 = np.sqrt( (vertexs[7][0] - vertexs[6][0])**2 + (vertexs[7][1] - vertexs[6][1])**2 )
    #         dist_8 = np.sqrt( (vertexs[0][0] - vertexs[7][0])**2 + (vertexs[0][1] - vertexs[7][1])**2 )
    #         # print "dist_1", dist_1, dist_2,dist_3, dist_4,dist_5, dist_6,dist_7, dist_8

    #         if abs(dist_1-dist_2)>0.5:
    #             possible_vertexs = [e for e in possible_vertexs if not np.array_equal(e,vertexs)]


    #     possible_centers=[]
    #     valid_vertexs=[]

    # #     print "circles", circle_coordinates
    #     for vertexs in possible_vertexs:
    #         # print "current vertexs", vertexs
    # #         print "vertexs", vertexs
    #         center_x = (sum([vertex[0] for vertex in vertexs])/len(vertexs)).astype(int)
    #         center_y = (sum([vertex[1] for vertex in vertexs])/len(vertexs)).astype(int)
    # #         print "center_x", (center_x, center_y)
    #         possible_centers.append((center_x, center_y))
    #         valid_vertexs.append(vertexs)

    #     centers=()
    #     vertexs=[]
    #     if len(possible_centers)>0:
    #         centers=possible_centers[0]
    #         vertexs=valid_vertexs[0]
    #     return centers, vertexs



        # print "detected edges", (edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8)


    #     horizontal_edges = sorted(coordinates, key=lambda x:abs(x[1]-0.0*np.pi/4.0))[:2]
    #     vertical_edges = sorted(vertical_lines, key=lambda x:x[0])
    #     edge3=vertical_edges[1]
    #     edge7=vertical_edges[0]

    #     vertical_edges = sorted(coordinates, key=lambda x:abs(x[1]-0.0*np.pi/4.0))[:2]
    #     vertical_edges = sorted(vertical_lines, key=lambda x:x[0])
    #     edge3=vertical_edges[1]
    #     edge7=vertical_edges[0]


    #     vertical_edges = sorted(coordinates, key=lambda x:abs(x[1]-0.0*np.pi/4.0))[:2]
    #     vertical_edges = sorted(vertical_lines, key=lambda x:x[0])
    #     edge3=vertical_edges[1]
    #     edge7=vertical_edges[0]

    #     right_edges = sorted(coordinates, key=lambda x:abs(x[1]-1.0*np.pi/4.0))[:2]
    #     right_edges = sorted(right_lines, key=lambda x:x[0])
    #     edge4=vertical_edges[1]
    #     edge8=vertical_edges[0]


    #     # print "vertical_lines", vertical_lines





        centers=()
        vertexs = [inter1, inter2, inter3, inter4, inter5, inter6, inter7, inter8]

        if len(vertexs)>0:
            center_x = (sum([vertex[0] for vertex in vertexs])/len(vertexs)).astype(int)
            center_y = (sum([vertex[1] for vertex in vertexs])/len(vertexs)).astype(int)
            # print center_x, center_y
            centers=(center_x, center_y)
        return centers, vertexs
    else:
        return [],[]

def find_square_sign(coordinates, threshold):
     
    """Part2: hough transform to detect circles
    """
#     # print "coordinates", coordinates  
   
    """
    Part3: Find the valid lines based on the theta difference
        Logic to use:
            1. The angle between two lines are 135
            2. The rho of current line with previous lines should have tangleable difference
            3. The vertexs are intersection of qualified lines
            4. The side lengh must be approximately close enough
            5. The centers BGR value should match the reference
    """
    
#     # print "total lines", len(coordinates)
    possible_edges = []
    for line1 in coordinates:
#         # print "#####start line1", line1
        for line2 in coordinates:
            
            if (line1!=line2) and abs(line1[1]-line2[1]-np.pi/2.0)<threshold:
#                 # print "@@@@@find line2", line1, line2
                for line3 in coordinates : 
                    
                    if (line3!=line2 and line3!=line1) and abs(abs(line2[1]-line3[1])-np.pi/2.0)<threshold \
                    and abs(line3[0]-line2[0])>5 and abs(line3[0]-line1[0])>5:
#                         # print "$$$$find line3", line1, line2, line3
                        for line4 in coordinates : 
                            
                            if (line4!=line3 and line4!=line2 and line4!=line1) and abs(abs(line3[1]-line4[1])-np.pi/2.0)<threshold \
                            and abs(line4[0]-line3[0])>5 and abs(line4[0]-line2[0])>5 and abs(line4[0]-line1[0])>5:
#                                 # print "%%%%find line4", line1, line2, line3, line4
                                possible_edges.append((line1, line2, line3, line4))
#                                 # print "coordinates before remove", coordinates
#                                 # print "current remove elements", line1[1], line2[1], line3[1], line4[1]
#                                 coordinates = [e for e in coordinates if e not in (line1, line2, line3, line4)]
# #                                 coordinates.remove(line1)
# #                                 coordinates.remove(line2)
# #                                 coordinates.remove(line3)
# #                                 coordinates.remove(line4)
#                                 # print "coordinates after remove", coordinates
                            else:
                                continue
                    else:
                        continue
            else:
                continue
#     # print possible_edges
#     edges=possible_edges[2]

     
    possible_vertexs=[]
    
    for edges in possible_edges:
        store_vertexs=[]
        for j in range(4):
            store_vertexs.append(intersec(edges[j-1], edges[j]))
#         # print "current store_vertexs", store_vertexs
        
        possible_vertexs.append(store_vertexs)
    
    
    
    for vertexs in possible_vertexs:
        # print "possing vertexs", vertexs
        dist_1 = np.sqrt( (vertexs[1][0] - vertexs[0][0])**2 + (vertexs[1][1] - vertexs[0][1])**2 )
        dist_2 = np.sqrt( (vertexs[2][0] - vertexs[1][0])**2 + (vertexs[2][1] - vertexs[1][1])**2 )
        # print "dist_1", dist_1
        # print "dist_2", dist_2
        if abs(dist_1-dist_2)>1:
            possible_vertexs = [e for e in possible_vertexs if not np.array_equal(e,vertexs)]
    
#     # print "possible_vertexs after distance check", possible_vertexs
    

    possible_centers=[]
    valid_vertexs=[]
    
#     print "circles", circle_coordinates
    for vertexs in possible_vertexs:
        # print "current vertexs", vertexs
#         print "vertexs", vertexs
        center_x = (sum([vertex[0] for vertex in vertexs])/len(vertexs)).astype(int)
        center_y = (sum([vertex[1] for vertex in vertexs])/len(vertexs)).astype(int)
#         print "center_x", (center_x, center_y)
        
#         A=img_in[center_y][center_x]  
#         B=[0,255,255]
        
#         if np.array_equal(A,B):
#             print "A", A, B
#             print "center", center_x, center_y
#             print "all circles", circle_coordinates
#             #QC Part, draw vertexs
#             for circle_coordinate in circle_coordinates:
#                 if abs(center_x - circle_coordinate[0]) <1.5 and abs(center_y - circle_coordinate[1]) <1.5:
            
            
        possible_centers.append((center_x, center_y))
        valid_vertexs.append(vertexs)
#             break 

#         else:
#             continue
            
    """Part4: return the outputs
    """    
    centers=()
    vertexs=[]
    if len(possible_centers)>0:
        centers=possible_centers[0]
        vertexs=valid_vertexs[0]
    return centers, vertexs

# def find_construction_sign(coordinates):
#     """Part2: hough transform to detect circles
#     """
# #     # print "coordinates", coordinates  
   
#     """
#     Part3: Find the valid lines based on the theta difference
#         Logic to use:
#             1. The angle between two lines are 135
#             2. The rho of current line with previous lines should have tangleable difference
#             3. The vertexs are intersection of qualified lines
#             4. The side lengh must be approximately close enough
#             5. The centers BGR value should match the reference
#     """
    
# #     # print "total lines", len(coordinates)
#     possible_edges = []
#     for line1 in coordinates:
# #         # print "#####start line1", line1
#         for line2 in coordinates:
            
#             if (line1!=line2) and abs(line1[1]-line2[1]-np.pi/2.0)<0.001:
# #                 # print "@@@@@find line2", line1, line2
#                 for line3 in coordinates : 
                    
#                     if (line3!=line2 and line3!=line1) and abs(abs(line2[1]-line3[1])-np.pi/2.0)<0.001 \
#                     and abs(line3[0]-line2[0])>5 and abs(line3[0]-line1[0])>5:
# #                         # print "$$$$find line3", line1, line2, line3
#                         for line4 in coordinates : 
                            
#                             if (line4!=line3 and line4!=line2 and line4!=line1) and abs(abs(line3[1]-line4[1])-np.pi/2.0)<0.001 \
#                             and abs(line4[0]-line3[0])>5 and abs(line4[0]-line2[0])>5 and abs(line4[0]-line1[0])>5:
# #                                 # print "%%%%find line4", line1, line2, line3, line4
#                                 possible_edges.append((line1, line2, line3, line4))
# #                                 # print "coordinates before remove", coordinates
# #                                 # print "current remove elements", line1[1], line2[1], line3[1], line4[1]
# #                                 coordinates = [e for e in coordinates if e not in (line1, line2, line3, line4)]
# # #                                 coordinates.remove(line1)
# # #                                 coordinates.remove(line2)
# # #                                 coordinates.remove(line3)
# # #                                 coordinates.remove(line4)
# #                                 # print "coordinates after remove", coordinates
#                             else:
#                                 continue
#                     else:
#                         continue
#             else:
#                 continue
# #     print possible_edges
# #     edges=possible_edges[2]

     
#     possible_vertexs=[]
    
#     for edges in possible_edges:
#         store_vertexs=[]
#         for j in range(4):
#             store_vertexs.append(intersec(edges[j-1], edges[j]))
# #         # print "current store_vertexs", store_vertexs
        
#         possible_vertexs.append(store_vertexs)
    
    
    
#     for vertexs in possible_vertexs:
#         # print "possing vertexs", vertexs
#         dist_1 = np.sqrt( (vertexs[1][0] - vertexs[0][0])**2 + (vertexs[1][1] - vertexs[0][1])**2 )
#         dist_2 = np.sqrt( (vertexs[2][0] - vertexs[1][0])**2 + (vertexs[2][1] - vertexs[1][1])**2 )
#         # print "dist_1", dist_1
#         # print "dist_2", dist_2
#         if abs(dist_1-dist_2)>1:
#             possible_vertexs = [e for e in possible_vertexs if not np.array_equal(e,vertexs)]
    
# #     print "possible_vertexs after distance check", possible_vertexs
    

#     possible_centers=[]
#     valid_vertexs=[]
    
# #     print "circles", circle_coordinates
#     for vertexs in possible_vertexs:
# #         print "current vertexs", vertexs
# #         print "vertexs", vertexs
#         center_x = (sum([vertex[0] for vertex in vertexs])/len(vertexs)).astype(int)
#         center_y = (sum([vertex[1] for vertex in vertexs])/len(vertexs)).astype(int)
# #         print "center_x", (center_x, center_y)
        
# #         A=img_in[center_y][center_x]  
# #         B=[0,128,255]
        
# #         if np.array_equal(A,B):
# #             print "A", A, B
# #             print "center", center_x, center_y
# #             print "all circles", circle_coordinates
# #             #QC Part, draw vertexs
# #             for circle_coordinate in circle_coordinates:
# #                 if abs(center_x - circle_coordinate[0]) <1.5 and abs(center_y - circle_coordinate[1]) <1.5:
            
            
#         possible_centers.append((center_x, center_y))
#         valid_vertexs.append(vertexs)
# #             break 

# #         else:
# #             continue
            
#     """Part4: return the outputs
#     """   
#     centers=()
#     vertexs=[]
#     if len(possible_centers)>0:
#         centers=possible_centers[0]
#         vertexs=valid_vertexs[0]
#     return centers, vertexs

def find_do_not_enter_sign(coordinates):
    """Part2: hough transform to detect circles
    """    
#     # print coordinates
    """Part3: Output the result
    """
    centers=()
    if len(coordinates)>0:

        center_x=coordinates[0][0]
        center_y=coordinates[0][1]
        # print (center_x, center_y)
        centers = (center_x, center_y) 
    return centers
   

def find_traffic_light(coordinates, circle_coordinates):
     
    """Part2: hough transform to detect circles
    """
        
#     # print "coordinates", coordinates  
   
    """
    Part3: Find the valid lines based on the theta difference
        Logic to use:
            1. The angle between two lines are 135
            2. The rho of current line with previous lines should have tangleable difference
            3. The vertexs are intersection of qualified lines
            4. The side lengh must be approximately close enough
            5. The centers BGR value should match the reference
    """
    
#     # print "total lines", len(coordinates)
    possible_edges = []
    for line1 in coordinates:
#         # print "#####start line1", line1
        for line2 in coordinates:
            
            if  (line2!=line1) and abs(line1[1]-line2[1]-np.pi/2.0)<0.00001:
#                 # print "@@@@@find line2", line1, line2
                for line3 in coordinates : 
                    
                    if (line3!=line2 and line3!=line1) and abs(abs(line2[1]-line3[1])-np.pi/2.0)<0.00001 \
                    and abs(line3[0]-line2[0])>5 and abs(line3[0]-line1[0])>5:
#                         # print "$$$$find line3", line1, line2, line3
                        for line4 in coordinates : 
                            
                            if (line4!=line3 and line4!=line2 and line4!=line1) and abs(abs(line3[1]-line4[1])-np.pi/2.0)<0.00001 \
                            and abs(line4[0]-line3[0])>5 and abs(line4[0]-line2[0])>5 and abs(line4[0]-line1[0])>5:
#                                 # print "%%%%find line4", line1, line2, line3, line4
                                possible_edges.append((line1, line2, line3, line4))
#                                 # print "coordinates before remove", coordinates
#                                 # print "current remove elements", line1[1], line2[1], line3[1], line4[1]
#                                 coordinates = [e for e in coordinates if e not in (line1, line2, line3, line4)]
# #                                 coordinates.remove(line1)
# #                                 coordinates.remove(line2)
# #                                 coordinates.remove(line3)
# #                                 coordinates.remove(line4)
#                                 # print "coordinates after remove", coordinates
                            else:
                                continue
                    else:
                        continue
            else:
                continue
#     # print possible_edges
#     edges=possible_edges[2]

     
    possible_vertexs=[]
    
    for edges in possible_edges:
        store_vertexs=[]
        for j in range(4):
            store_vertexs.append(intersec(edges[j-1], edges[j]))
#         # print "current store_vertexs", store_vertexs
        possible_vertexs.append(store_vertexs)
    
    for vertexs in possible_vertexs:
        # print "possing vertexs", vertexs
        dist_1 = np.sqrt( (vertexs[1][0] - vertexs[0][0])**2 + (vertexs[1][1] - vertexs[0][1])**2 )
        dist_2 = np.sqrt( (vertexs[2][0] - vertexs[1][0])**2 + (vertexs[2][1] - vertexs[1][1])**2 )
        dist_3 = np.sqrt( (vertexs[3][0] - vertexs[2][0])**2 + (vertexs[3][1] - vertexs[2][1])**2 )
        dist_4 = np.sqrt( (vertexs[0][0] - vertexs[3][0])**2 + (vertexs[0][1] - vertexs[3][1])**2 )
        # print "dist_1", dist_1
        # print "dist_2", dist_2
        if abs(dist_1-dist_2)<1 or abs(dist_3-dist_4)<1 or abs(dist_1-dist_3)>1 or abs(dist_2-dist_4)>1 :
            possible_vertexs = [e for e in possible_vertexs if not np.array_equal(e,vertexs)]
    
#     # print "possible_vertexs after distance check", possible_vertexs
    
    possible_centers=[]
    valid_vertexs=[]
    
    for vertexs in possible_vertexs:
        # print "current vertexs", vertexs
    
        center_x = (sum([vertex[0] for vertex in vertexs])/len(vertexs)).astype(int)
        center_y = (sum([vertex[1] for vertex in vertexs])/len(vertexs)).astype(int)
        
#         possible_centers.append((center_x, center_y))
#         valid_vertexs.append(vertexs)
#         print "line centers", center_x, center_y 
#         print "circle_coordinates", circle_coordinates
        for circle_coordinate in circle_coordinates:
            if abs(center_x - circle_coordinate[0]) <3 and abs(center_y - circle_coordinate[1]) <3:
                
                possible_centers.append((center_x, center_y))
                valid_vertexs.append(vertexs)
#                 break
            else:
                continue

    """Part4: return the outputs
    """ 
    centers=()
    vertexs=[]
    if len(possible_centers)>0:
        centers=possible_centers[0]
        vertexs=valid_vertexs[0]

    return centers, vertexs

def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    
    
#Part1: preprocessing img_in
    
#     print img_in.shape
#     print img_in[390][287]
    gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    result = img_in.copy() 

#     # print (1.*np.min(gray), 1.*np.max(gray), np.mean(gray), np.std(gray)) 
   
    #hough transform

#Part2: hough transform to detect circles    
    min_r=radii_range[0]
    max_r=radii_range[-1]+1
#     # print min_r, max_r

    coordinates=houghcircles(gray, 50, min_r, max_r)  
#     # print coordinates
#     for i in range(len(coordinates)):
#         # print i 
#         a=coordinates[i]
#         b=coordinates[:i] + coordinates[i+1 :]
#         # print a
#         # print b 
#     a = np.random.normal(size=(10,3))
#     b = np.random.normal(size=(1,3))
    
#     # print coordinates

# based on y(vertical) distance of circle, remove outliers
    if abs(coordinates[0][0]-coordinates[1][0]) > 5:
        coordinates=coordinates[1:]
        
    if abs(coordinates[-1][0]-coordinates[-2][0])>5:
        coordinates=coordinates[:-1]
    
    
    coordinates = sorted(coordinates, key=lambda x: x[1])
    coordinate = coordinates[1][0:-1]
#     print coordinate
        
    
# Part4: based on the HLS color space, luma, to select the lightest circle
# based on vertical distance of circle, assign state
    hls = cv2.cvtColor(img_in, cv2.COLOR_BGR2HLS)
#     # print hsv.shape 
#     # print "green",img_in[364][287]
#     # print "red",img_in[44][283]
#     # print "yellow",img_in[299][134]
    
#     # print "green",hls[364][287]
#     # print "red",hls[44][283]
#     # print "yellow",hls[299][283]    
#     # print coordinates
    
    indec=9
    luma=0
    for k in range(len(coordinates)):
        x=coordinates[k][0]
        y=coordinates[k][1]
        
        hls_value=hls[y][x]
        if hls[y][x][1] > luma:
            luma=hls[y][x][1]
            indec = k 
#     # print indec    
        
#Part5: output the required result   
    
#     # print luma, coordinate, hue
    detect_light=()
    if indec==0:
        detect_light=(coordinate, 'red')
    if indec==1:
        detect_light=(coordinate, 'yellow')
    if indec==2:
        detect_light=(coordinate, 'green')
        
#     print 'coord', coordinates[1] 
    
#     cv2.circle(result,(coordinates[1][0],coordinates[1][1]),coordinates[1][2],(255,0,255),2) 
# #         # draw the center of the circle 
#     cv2.circle(result,(coordinates[1][0],coordinates[1][1]),2,(255,0,255),3) 
#     cv2.imshow('Result', result)
#     cv2.waitKey(0)

      
        
#         green=np.array([0,255,0])
#         red=np.array([0,0,255])
#         yellow=np.array([255,255,0])
#         if np.array_equal(color,green):
#             result=('green',(x,y))
#         if np.array_equal(color,red):
#             result=('red',(x,y))
#         if np.array_equal(color,yellow):
#             result=('yellow',(x,y))

#     print detect_light
    return detect_light 
            
        
    
#     def hsv(blue, green, red):
#         color=np.uint8([[[blue, green, red]]])
#         hsv_color=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
#         hue=hsv_color[0][0][0]
#         hsv_lower = np.array([hue-10, 100, 100], dtype=np.uint8)
#         hsv_upper = np.array([hue+10, 255, 255], dtype=np.uint8)
#         return hsv_lower, hsv_upper 
    
#     green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
    
#     green=np.uint8([[[0,255,0]]])
#     red=np.unit8([[[0,0,255]]])
#     yellow=np.uint8([[[255,255,0]]])
    
    
    
#     raise NotImplementedError

def traffic_light_scene_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    
    """Part1: Preprocessin the image
    """
#     print img_in[250][300]
#     img_in[np.where((img_in!=[0,255,255]).all(axis=2))] = [0,204,0]
    img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    
#     print "img_hsv", img_hsv[113][538]
    lower = np.array([0,0,46]) #example value
    upper = np.array([180,43,100]) #example value
    mask = cv2.inRange(img_hsv, lower, upper)
#     mask=generate_mask(img_in, [51,51,51])
    edges=generate_edges(mask)
#     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred,100,221)
    result = img_in.copy() 
    """
    Part2: Hough Transform to generate the lines 
    """
#     ret,thresh = cv2.threshold(gray,127,255,0)
    coordinates=houghline(edges, 27) 
    
    circle_coordinates=houghcircles(edges,50,0,50)
# #     print coordinates
    
#     """Part3: Output the result
#     """
#     if len(coordinates)>0:
#         circles_check(result, coordinates)
    
    # cv2.imshow('img_in', img_in)
    # cv2.imshow('mask', mask)
#     cv2.imshow('gray', gray)
#     cv2.imshow('blurred', blurred)
#     cv2.imshow('edges', edges)
    # lines_check(result, coordinates)
    # circles_check(result, circle_coordinates)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    #
    
    if len(coordinates) > 0 and len(circle_coordinates):
        centers,vertexs = find_traffic_light(coordinates, circle_coordinates)
#         print "centers", centers
        """
        Part3: Output the result 
        """
        if len(centers) >0:
            # for i in range(len(vertexs)):
            #     cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
            #
            # lines_check(result, coordinates)
            # circles_check(result, circle_coordinates)
            #
            # cv2.imshow('Result', result)
            # cv2.waitKey(0)
            return centers
    
def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    
    """
    Part1: Preprocessing the image
    """
    # print img_in[173][358]
    
#     lower_blue=np.array([0,43,46])
#     upper_blue=np.array([10,255,255])
#     hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     res = cv2.bitwise_and(img_in, img_in, mask=mask)
#     img_piece=img_in[np.where((img_in==[0,0,255]).all(axis=2))]        
#     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,221]) #example value
    upper = np.array([180,30,255]) #example value
    mask = cv2.inRange(img_hsv, lower, upper)
#     mask=generate_mask(img_in, [255,255,255])
    edges=generate_edges(mask)

    
    """
    Part2: Hough Transform to generate the lines 
    """
#     ret,thresh = cv2.threshold(gray,127,255,0)
    coordinates=houghline(edges, 40)
    
    # cv2.imshow('img_in', img_in)
    # cv2.imshow('mask', mask)
#     cv2.imshow('gray', gray)
#     cv2.imshow('blurred', blurred)
#     cv2.imshow('edges', edges)
#     cv2.waitKey(0)
    result=img_in.copy()
    
#     print "lenght", len(coordinates)
    
    if len(coordinates) > 0:
#         if find_yield_sign(coordinates) is not None:
        centers, vertexs=find_yield_sign(coordinates)

        if len(centers)>0:
            # for i in range(len(vertexs)):
            #         cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
            # lines_check(result, coordinates)
            #
            # cv2.imshow('Result', result)
            # cv2.waitKey(0)
            return centers

#     raise NotImplementedError


    

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    
    """
    Part1: Preprocessing the image
    """
    # print img_in[249][149]
#     img_in[np.where((img_in==[0,0,204]).all(axis=2))] = [255,0,204]
#     img_in[np.where((img_in==[255,255,255]).all(axis=2))] = [0,0,204]
#     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred,100,221)
#     lines = cv2.HoughLines(edges,1,np.pi/180,32)
    
#     print img_in[220][149]

    img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([0,43,46]) #example value
    upper = np.array([10,255,230]) #example value
    mask = cv2.inRange(img_hsv, lower, upper)
    
#     mask=generate_mask(img_in, [0,0,204])
    
#     print "mask", mask
       
    edges=generate_edges(mask)
    coordinates=houghline(edges,25)
    
    
#     circles_check(result, coordinates)
#     print coordinates
    result=img_in.copy()
    
#     cv2.imshow('img_in', img_in)
#     cv2.imshow('mask', mask)
#     cv2.imshow('edges', edges)
#
# #     cv2.imshow('gray', gray)
# #     cv2.imshow('blurred', blurred)
#     lines_check(result, coordinates)
#     cv2.imshow('Result', result)
#     cv2.waitKey(0)
    
    
    
    if len(coordinates) > 0:
        centers,vertexs = find_stop_sign(coordinates,30)
        if len(centers) > 0:
            # for i in range(len(vertexs)):
            #     cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
            #     text="n{}".format(i)
            #     cv2.putText(result, text, (vertexs[i][0],vertexs[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 2)
            #
            # lines_check(result, coordinates)
            #
            # cv2.imshow('Result', result)
            # cv2.waitKey(0)

            return centers

#     raise NotImplementedError


    
    
    
def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    
    """Part1: Preprocessin the image
    """
#     print img_in[250][300]
#     img_in[np.where((img_in!=[0,255,255]).all(axis=2))] = [0,204,0]
    img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([26,43,46]) #example value
    upper = np.array([34,255,255]) #example value
    mask = cv2.inRange(img_hsv, lower, upper)
#     mask=generate_mask(img_in, [0,255,255])
    edges=generate_edges(mask)
#     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred,100,221)
    result = img_in.copy() 
    """
    Part2: Hough Transform to generate the lines 
    """
#     ret,thresh = cv2.threshold(gray,127,255,0)
    coordinates=houghline(edges, 40) 
    
    # cv2.imshow('img_in', img_in)
    # cv2.imshow('mask', mask)
#     cv2.imshow('gray', gray)
#     cv2.imshow('blurred', blurred)
#     cv2.imshow('edges', edges)
    result=img_in.copy()
    
    if len(coordinates) > 0:
        centers,vertexs = find_square_sign(coordinates, 0.001)

        """
        Part3: Output the result 
        """
        if len(centers) >0:
            # for i in range(len(vertexs)):
            #     cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
            #
            # lines_check(result, coordinates)
            # cv2.imshow('Result', result)
            # cv2.waitKey(0)

            return centers



    
def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    
    """Part1: Preprocessin the image
    """
#     # print img_in[150][200]
#     [  0 128 255]
#     img_in[np.where((img_in==[0,255,255]).all(axis=2))] = [0,255,0]
#     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred,100,221)

    img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([11,43,46]) #example value
    upper = np.array([25,255,255]) #example value
    mask = cv2.inRange(img_hsv, lower, upper)
    
#     mask=generate_mask(img_in, [0,128,255])
    edges=generate_edges(mask)
#     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred,100,221)
    result = img_in.copy() 
    """
    Part2: Hough Transform to generate the lines 
    """
#     ret,thresh = cv2.threshold(gray,127,255,0)
    coordinates=houghline(edges, 32) 
    
    # cv2.imshow('img_in', img_in)
    # cv2.imshow('mask', mask)
    # cv2.imshow('gray', gray)
    # cv2.imshow('blurred', blurred)
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)

    
    if len(coordinates) > 0:
        centers,vertexs = find_square_sign(coordinates, 0.001)

        """
        Part3: Output the result 
        """
        if len(centers) >0:
            # for i in range(len(vertexs)):
            #     cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
            #
            # lines_check(result, coordinates)
            # cv2.imshow('Result', result)
            # cv2.waitKey(0)

            return centers

#     raise NotImplementedError


    

def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    
    """Part1: preprocessing img_in
    """
#     print img_in[120][145]
#     img_in[np.where((img_in==[255,255,255]).all(axis=2))] = [0,0,255]
#     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred,100,221)

    img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([0,43,230]) #example value
    upper = np.array([10,255,255]) #example value
    mask = cv2.inRange(img_hsv, lower, upper)

#     mask=generate_mask(img_in, [0,0,255])
    edges=generate_edges(mask)
    
    # cv2.imshow('img_in', img_in)
#     cv2.imshow('gray', gray)
#     cv2.imshow('blurred', blurred)
#     cv2.imshow('mask', mask)
#     cv2.imshow('edges', edges)
    
    result = img_in.copy() 
    
#     coordinates=houghline(edges, 32) 
    
#     centers,vertexs = find_construction_sign(coordinates)
    
#     """
#     Part3: Output the result 
#     """
    
#     for i in range(len(vertexs)):
#         cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2) 
    
    coordinates=houghcircles(edges,300,25,50)
#     print coordinates
    
    """Part3: Output the result
    """
    if len(coordinates)>0:
        # circles_check(result, coordinates)
        centers_r = find_do_not_enter_sign(coordinates)
        if len(centers_r)>0:
            x = centers_r[0]
            y = centers_r[1]
            centers = (x,y)
            # cv2.imshow('Result', result)
            # cv2.waitKey(0)

            return centers
        

 

    
#     raise NotImplementedError


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    
    """Part1: Preprocessin the image
    """  
#     temp_img=img_in.copy()
#     temp_img[np.where((img_in!=[0,255,255]).all(axis=2))] = [0,255,0]
#     gray = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred,100,250)
#     print img_in[199][100]
#     signs = ['traffic_light', 'no_entry', 'stop', 'warning', 'yield', 'construction']
    
#     result=img_in.copy()
#     mask=generate_mask(img_in, [51,51,51])
#     edges=generate_edges(mask)
#     coordinates=houghcircles(edges,50,0,50)
#     coordinates = sorted(coordinates, key=lambda x: x[1])
#     circles_check(result, coordinates)
#     coordinate = coordinates[0]
# #     print coordinate
#     result['traffic_light'] = coordinate
    
#     cv2.imshow('img_in', img_in)
# #     cv2.imshow('gray', gray)
# #     cv2.imshow('blurred', blurred)
#     cv2.imshow('mask', mask)
#     cv2.imshow('edges', edges)
#     cv2.imshow('result', result)
#     cv2.waitKey(0)
    objects=dict()
    
#     x_traffic_light, y_traffic_light=find_traffic_light(edges)
#
#     x_stop, y_stop = find_stop(edges)
#     x_warning, y_warning = find_warning_sign(edges)
#     x_yield, y_yield = find_yield(edges)
#     x_construction, y_construction = find_construction(edges)
    objects['traffic_light'] = traffic_light_scene_detection(img_in)
    objects['no_entry'] = do_not_enter_sign_detection(img_in)
    objects['stop'] = stop_sign_detection(img_in)
    objects['warning'] = warning_sign_detection(img_in)
    objects['yield'] = yield_sign_detection(img_in)
    objects['construction'] = construction_sign_detection(img_in)
  
    
    objects = {i:j for i,j in objects.items() if j is not None}
    # print "objects", objects
    return objects
    
#     return signs

    
    
    
#     raise NotImplementedError


# objects {'no_entry': (235, 335), 'stop': (348, 347), 'yield': (506, 331), 
#          'traffic_light': (113, 338), 'warning': (799, 347), 'construction': (649, 347)}

def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    
    
    
#     print "yield center",  img_in[235][335], img_in[348][347], img_in[506][331],  img_in[113][338]
    blurred = cv2.GaussianBlur(img_in, (5, 5), 0)
    denoised=cv2.fastNlMeansDenoisingColored(blurred,None,10,10,7,21)
#     print "blurred center", blurred[235][335], blurred[348][347], blurred[506][331], blurred[113][338] 
    img_hsv=cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
#     lower_red = np.array([11,50,50]) #example value
#     upper_red = np.array([25,255,255]) #example value
#     mask = cv2.inRange(img_hsv, lower_red, upper_red)
#     edges=generate_edges(mask)

#         hue=hsv_color[0][0][0]
#         hsv_lower = np.array([hue-10, 100, 100], dtype=np.uint8)
#         hsv_upper = np.array([hue+10, 255, 255], dtype=np.uint8)
#         return hsv_lower, hsv_upper 
    
#     green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
#     cv2.imshow('edges', edges)    
#     cv2.imshow('mask', mask)
    
    objects=dict()
#     img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    
    signs = ['traffic_light', 'no_entry', 'stop', 'warning', 'yield', 'construction']
    for sign in signs:
        if sign=='traffic_light':
            # img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
            lower = np.array([26,43,46]) #example value
            upper = np.array([34,255,255]) #example value
            mask = cv2.inRange(img_hsv, lower, upper)
            edges=generate_edges(mask)

            # cv2.imshow('img_in', img_in)
        #     cv2.imshow('gray', gray)
        #     cv2.imshow('blurred', blurred)
        #     cv2.imshow('mask', mask)
        #     cv2.imshow('edges', edges)
        #     cv2.waitKey(0)

            result = img_in.copy()

            coordinates=houghcircles(edges,400,10,18)
        #     print coordinate
            if len(coordinates)>0:
                # circles_check(result, coordinates)
                x = coordinates[0][0]
                y = coordinates[0][1]
                centers = (x,y)
                # cv2.imshow('Result', result)
                # cv2.waitKey(0)
                objects['traffic_light'] = centers

            # objects['traffic_light'] = traffic_light_scene_detection(blurred)
        elif sign=='no_entry':
            # img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
            lower = np.array([0,43,230]) #example value
            upper = np.array([10,255,255]) #example value
            mask = cv2.inRange(img_hsv, lower, upper)
            edges=generate_edges(mask)

            # cv2.imshow('img_in', img_in)
        #     cv2.imshow('gray', gray)
        #     cv2.imshow('blurred', blurred)
        #     cv2.imshow('mask', mask)
        #     cv2.imshow('edges', edges)
        #     cv2.waitKey(0)

            result = img_in.copy()
            coordinates=houghcircles(edges,600,25,40)
        #    """
            if len(coordinates)>0:
                # circles_check(result, coordinates)
                x = coordinates[0][0]
                y = coordinates[0][1]
                centers = (x,y)
                # cv2.imshow('Result', result)
                # cv2.waitKey(0)
                objects['no_entry'] = centers

            # objects['no_entry'] = do_not_enter_sign_detection(blurred)
            
        elif sign=='stop':
            # img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
            lower = np.array([0,43,46]) #example value
            upper = np.array([10,255,230]) #example value
            mask = cv2.inRange(img_hsv, lower, upper)
            edges=generate_edges(mask)
            coordinates=houghline(edges,30)
            result=img_in.copy()
        #     cv2.imshow('img_in', img_in)
        #     cv2.imshow('blurred', blurred)
        #     cv2.imshow('mask', mask)
        # #     cv2.imshow('gray', gray)
        # #     cv2.imshow('blurred', blurred)
        #     cv2.imshow('edges', edges)
        #     lines_check(result, coordinates)
        #     cv2.imshow('Result', result)
        #     cv2.waitKey(0)
            if len(coordinates) > 0:
                centers,vertexs = find_stop_sign(coordinates,30)
                if len(centers) > 0:
                    # for i in range(len(vertexs)):
                    #     cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
                    #     text="{}".format(i+1)
                    #     cv2.putText(result, text, (vertexs[i][0],vertexs[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 2)
                    # cv2.imshow('Result', result)
                    # cv2.waitKey(0)
                    objects['stop'] = centers
        elif sign=='warning':
            objects['warning'] = warning_sign_detection(denoised)
        elif sign=='yield':
            objects['yield'] = yield_sign_detection(denoised)
        elif sign=='construction':
            objects['construction'] = construction_sign_detection(denoised)
        else:
            break
        
        

    
#     objects['traffic_light'] = traffic_light_scene_detection(blurred)
#     objects['no_entry'] = do_not_enter_sign_detection(blurred)
#     objects['stop'] = stop_sign_detection(denoised)
#     objects['warning'] = warning_sign_detection(blurred)
#     objects['yield'] = yield_sign_detection(blurred)
#     objects['construction'] = construction_sign_detection(blurred)
  
    
    objects = {i:j for i,j in objects.items() if j is not None}
    # print "objects", objects
    return objects
    
    
#     raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    # print img_in.shape
    blurred = cv2.GaussianBlur(img_in, (5, 5), 0)
    # denoised=cv2.fastNlMeansDenoisingColored(blurred,None,10,10,7,21)
#     print "blurred center", blurred[235][335], blurred[348][347], blurred[506][331], blurred[113][338]


    # cv2.imshow('img_in', img_in)
    #     cv2.imshow('gray', gray)
    # cv2.imshow('blurred', blurred)
    #     cv2.imshow('mask', mask)
    # cv2.imshow('denoised', denoised)

    # cv2.waitKey(0)

    objects=dict()
#     img_hsv=cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)

    signs = ['traffic_light', 'no_entry', 'stop', 'warning', 'yield', 'construction']
    for sign in signs:
        # print "current sign", sign
        if sign=='traffic_light':
            img_hsv=cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            lower = np.array([26,43,46]) #example value
            upper = np.array([34,255,255]) #example value
            mask = cv2.inRange(img_hsv, lower, upper)
            edges=generate_edges(mask)
            # cv2.imshow('img_in', img_in)
        #     cv2.imshow('gray', gray)
        #     cv2.imshow('blurred', blurred)
        #     cv2.imshow('mask', mask)
        #     cv2.imshow('edges', edges)
        #     cv2.waitKey(0)

            result = img_in.copy()

            coordinates=houghcircles(edges,2000,0,1)
        #     print coordinate
            if len(coordinates)>0:
                # circles_check(result, coordinates)
                x = coordinates[0][0]
                y = coordinates[0][1]
                centers = (x,y)
                # cv2.imshow('Result', result)
                # cv2.waitKey(0)
                objects['traffic_light'] = centers

            # objects['traffic_light'] = traffic_light_scene_detection(blurred)
        elif sign=='no_entry':
            img_hsv=cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # print img_hsv.shape
            # print img_hsv[231][448]
            lower = np.array([156,43,200]) #example value
            upper = np.array([180,255,255]) #example value
            mask0 = cv2.inRange(img_hsv, lower, upper)
            lower = np.array([0,43,200]) #example value
            upper = np.array([10,100,255]) #example value
            mask1 = cv2.inRange(img_hsv, lower, upper)
            mask=mask0+mask1

            edges=generate_edges(mask)

            # cv2.imshow('img_in', img_in)
        #     cv2.imshow('gray', gray)
        #     cv2.imshow('blurred', blurred)
        #     cv2.imshow('mask', mask)
        #     cv2.imshow('edges', edges)
        #     cv2.waitKey(0)

            result = img_in.copy()
            circles = cv2.HoughCircles(edges,cv2.cv.CV_HOUGH_GRADIENT,1,2000,param1=80,param2=30,minRadius=50,maxRadius=130)
            coordinates=[]
            if circles is not None:
                circles=circles.astype(int)
                for i in circles[0,:]:
                    x=i[0]
                    y=i[1]
                    r=i[2]
                    coordinates.append((x,y,r))
                    coordinates = sorted(coordinates, key=lambda x: x[0])

            # coordinates=houghcircles(edges,800,50,130)
            # circles_check(result, coordinates)
            # cv2.imshow('Result', result)
            # cv2.waitKey(0)
        #    """
            if len(coordinates)>0:
                x = coordinates[0][0]
                y = coordinates[0][1]
                centers = (x,y)
                objects['no_entry'] = centers
        #     # objects['no_entry'] = do_not_enter_sign_detection(blurred)
        #
        elif sign=='stop':
            # print img_in.shape
            img_hsv=cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            lower = np.array([0,43,46]) #example value
            upper = np.array([10,255,230]) #example value
            mask = cv2.inRange(img_hsv, lower, upper)
            edges=generate_edges(mask)
            coordinates=houghline(edges,80)
            # result=img_in.copy()
            # cv2.imshow('img_in', img_in)
            # cv2.imshow('blurred', blurred)
            # cv2.imshow('mask', mask)
        #     cv2.imshow('gray', gray)
        #     cv2.imshow('blurred', blurred)
        #     cv2.imshow('edges', edges)
        #     lines_check(result, coordinates)
        #     cv2.imshow('Result', result)
        #     cv2.waitKey(0)
            if len(coordinates) > 0:
                centers,vertexs = find_stop_sign(coordinates,30)
                if len(centers) > 0:
                    # for i in range(len(vertexs)):
                    #     cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
                    #     text="{}".format(i+1)
                    #     cv2.putText(result, text, (vertexs[i][0],vertexs[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 2)
                    # cv2.imshow('Result', result)
                    # cv2.waitKey(0)
                    objects['stop'] = centers
        elif sign=='warning':
            img_hsv=cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            lower = np.array([26,43,46]) #example value
            upper = np.array([34,255,255]) #example value
            mask = cv2.inRange(img_hsv, lower, upper)

        #     mask=generate_mask(img_in, [0,128,255])
            edges=generate_edges(mask)
        #     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
        #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #     edges = cv2.Canny(blurred,100,221)
            result = img_in.copy()
        # #     ret,thresh = cv2.threshold(gray,127,255,0)
            coordinates=houghline(edges, 80)
        #     # cv2.imshow('img_in', img_in)
        #     # cv2.imshow('mask', mask)
        #     # cv2.imshow('gray', gray)
        #     # cv2.imshow('blurred', blurred)
        #     # cv2.imshow('edges', edges)
        #     # cv2.waitKey(0)
            if len(coordinates) > 0:
                centers,vertexs = find_square_sign(coordinates, 0.1)
                if len(centers) >0:
                    # for i in range(len(vertexs)):
        #             #     cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
        #             # lines_check(result, coordinates)
        #             # cv2.imshow('Result', result)
        #             # cv2.waitKey(0)
                    objects['warning'] = centers
        elif sign=='yield':
            img_hsv=cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # print img_hsv[150][340]
            lower = np.array([156,43,46]) #example value
            upper = np.array([180,255,255]) #example value
            mask = cv2.inRange(img_hsv, lower, upper)
        #     mask=generate_mask(img_in, [255,255,255])
            edges=generate_edges(mask)
            # cv2.imshow('mask', mask)
            # cv2.imshow('edges', edges)
            # result=img_in.copy()
            coordinates=houghline(edges, 80)
            # lines_check(result, coordinates)
            # cv2.imshow('lines', result)
            # cv2.waitKey(0)
            #
        #     cv2.imshow('gray', gray)
        #     cv2.imshow('blurred', blurred)
        #     cv2.waitKey(0)
        #     print "lenght", len(coordinates)
            if len(coordinates) > 0:
        #         if find_yield_sign(coordinates) is not None:
                centers, vertexs=find_yield_sign(coordinates)
                if len(centers)>0:
                    # for i in range(len(vertexs)):
                    #         cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
                    # lines_check(result, coordinates)

                    # cv2.imshow('Result', result)
                    # cv2.waitKey(0)
                    objects['yield'] = centers
        elif sign=='construction':
            # print "hsv", img_hsv[240][800]
            img_hsv=cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            lower = np.array([0,43,46]) #example value
            upper = np.array([10,255,255]) #example value
            mask = cv2.inRange(img_hsv, lower, upper)
        #     mask=generate_mask(img_in, [0,128,255])
            edges=generate_edges(mask)
        #     gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
        #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #     edges = cv2.Canny(blurred,100,221)
        #     result = img_in.copy()
        #     ret,thresh = cv2.threshold(gray,127,255,0)
            coordinates=houghline(edges, 100)

            # cv2.imshow('img_in', img_in)
            # cv2.imshow('mask', mask)
            # cv2.imshow('gray', gray)
            # cv2.imshow('blurred', blurred)
            # cv2.imshow('edges', edges)
            # lines_check(result, coordinates)
            # cv2.imshow('lines', result)
            # cv2.waitKey(0)
            if len(coordinates) > 0:
                centers,vertexs = find_square_sign(coordinates, 0.1)
                # print "centers", centers
                if len(centers) >0:
                    # for i in range(len(vertexs)):
                    #     cv2.circle(result,(vertexs[i][0],vertexs[i][1]),20,(0,112,255),2)
                    # lines_check(result, coordinates)
                    # cv2.imshow('Result', result)
                    # cv2.waitKey(0)
                    objects['construction'] = centers
        else:
            break

    objects = {i:j for i,j in objects.items() if j is not None}
    # print "objects", objects
    return objects
    
    # raise NotImplementedError
# simple_tl', 'scene_tl_1', 'scene_tl_2', 'scene_tl_3

# if __name__ == '__main__':
#     img_in = cv2.imread("input_images/img-5-b-1.png")
#     radii_range=range(10,30,1)
#     traffic_light_detection(img_in, radii_range)
#     yield_sign_detection(img_in)    
#     construction_sign_detection(img_in)
#     stop_sign_detection(img_in)
#     warning_sign_detection(img_in)
#     do_not_enter_sign_detection(img_in)
#     traffic_sign_detection(img_in)
#     traffic_sign_detection_noisy(img_in)
#     traffic_sign_detection_challenge(img_in)