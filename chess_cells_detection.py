import cv2
import numpy as np
import math
import glob
import sys
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

def segment_by_angle_kmeans(lines, k=2, **kwargs):

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    lines1 = []

    for line in lines:
      x = line[0][1]
      if 0<x<math.pi/12 or math.pi/12*5<x<math.pi/12*7 or math.pi/12*11<x<math.pi:
        lines1.append(line)

    lines = lines1


    # Get angles in [0, pi] radians
    angles = np.array([line[0][1] for line in lines])

    #angles = [0<x<math.pi/12 or math.pi/12*5<x<math.pi/12*7 or math.pi/12*11<x<math.pi for x in angles]

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)


    # Run k-means
    if sys.version_info[0] == 2:
        # python 2.x
        ret, labels, centers = cv2.kmeans(pts, k, criteria, attempts, flags)
    else: 
        # python 3.x, syntax has changed.
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

    labels = labels.reshape(-1) # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())
    print("Segmented lines into two groups: %d, %d" % (len(segmented[0]), len(segmented[1])))

    return segmented

def segmented_intersections(lines):
  """Finds the intersections between groups of lines."""

  intersections = []
  for i, group in enumerate(lines[:-1]):
    for next_group in lines[i+1:]:
      for line1 in group:
        for line2 in next_group:
          intersections.append(intersection(line1, line2)) 

  return intersections

def intersection(line1, line2):
  """Finds the intersection of two lines given in Hesse normal form.

  Returns closest integer pixel locations.
  See https://stackoverflow.com/a/383527/5087436
  """
  rho1, theta1 = line1[0]
  rho2, theta2 = line2[0]
  A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
  b = np.array([[rho1], [rho2]])
  x0, y0 = np.linalg.solve(A, b)
  x0, y0 = int(np.round(x0)), int(np.round(y0))
  return [[x0, y0]]

def grouping_and_mean(data, maxgap):

  values__ = []
  data.sort()
  groups = [[data[0]]]

  for x in data[1:]:
    if abs(x - groups[-1][-1]) <= maxgap:
      groups[-1].append(x)
    else:
      groups.append([x])
  #print(groups)

  for i in range(len(groups)):
    values = groups[i]
    average = sum(values)/len(values)
    values__.append(int(average))
  
  return values__

def drawLines(img, lines, color=(0,0,255)):

  for line in lines:
    for rho,theta in line:
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 2000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 2000*(-b))
      y2 = int(y0 - 1000*(a))
      cv2.line(img, (x1,y1), (x2,y2), color, 1)


class detectChessBoard:

  def __init__(self) -> None:
      pass      

  def calibration():

    # Defining the dimensions of checkerboard
    CHECKERBOARD = (7,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob('/home/sarath/calibration_new/*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray,(5,5),0)
        _, img_binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #cv2.imshow('img',gray)
        #cv2.waitKey(0)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(img_binary, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        #cv2.imshow('img',img)
        #cv2.waitKey(0)

    cv2.destroyAllWindows()

    h,w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    #print(h,w)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


  def group_split(arr, size):
      arrs = []
      while len(arr) > size:
          pice = arr[:size]
          arrs.append(pice)
          arr   = arr[size:]
      arrs.append(arr)
      return arrs

  def grouping(data, maxgap):

    values__ = []
    data.sort()
    groups = [[data[0]]]

    for x in data[1:]:
      if abs(x - groups[-1][-1]) <= maxgap:
        groups[-1].append(x)
      else:
        groups.append([x])

    return groups

  def get_chess_cells_array(x_points_, y_points_):

    cell_complete = []

    # outer loop
    for i in range(8):
      # inner loop
      for j in range(8):
          cell_name = (chr(72-j)+chr(56-i))
          cell_top_left_point = (x_points_[i][j],y_points_[i][j])
          cell_bottom_right_point = (x_points_[i+1][j+1],y_points_[i+1][j+1])
          cell_single = [cell_name , cell_top_left_point, cell_bottom_right_point]        
          cell_complete.append(cell_single)
          
    return cell_complete

  def get_cell_name(test_point,cell_full,wh):
    max = 0
    name = ''
    for i,x in enumerate(cell_full):
      w = wh[0]
      h = wh[1]
      mask_1 = np.zeros((w,h))
      mask_x = np.ones((abs(x[1][0]-x[2][0]),abs(x[1][1]-x[2][1])))
      mask_1[x[1][0]:x[2][0], x[1][1]:x[2][1]] = mask_x

      mask_2 = np.zeros((w,h))
      mask_tp = np.ones((abs(test_point[0][0][0]-test_point[1][0][0]),abs(test_point[0][0][1]-test_point[1][0][1])))
      mask_2[test_point[0][0][0]:test_point[1][0][0], test_point[0][0][1]:test_point[1][0][1]] = mask_tp

      result = np.add(mask_1, mask_2)
      sum = np.count_nonzero(result==2)

      if sum > max:
        max = sum
        name = x[0]

    return name

  def split_cells(x_points_split_ , y_points_split_):
    cells = []
    # outer loop
    for i in range(len(x_points_split_)-1):
      # inner loop
      for j in range(len(y_points_split_) -1):
          cells.append([(x_points_split_[i],y_points_split_[j]), (x_points_split_[i+1],y_points_split_[j+1])])
    return cells

  def check_multiple_cell(test_point1, x_points1, y_points1):  

    x_points_split = [test_point1[0][0] , test_point1[1][0]]
    y_points_split = [test_point1[0][1] , test_point1[1][1]]

    for i in range(8):
      if x_points1[i] >= test_point1[0][0] and x_points1[i] <= test_point1[1][0]:
        x_points_split.append(x_points1[i])
        #print(x_points1[i])

    for i in range(8):
      if y_points1[i] >= test_point1[0][1] and y_points1[i] <= test_point1[1][1]:
        y_points_split.append(y_points1[i])
        #print(y_points1[i])

    x_points_split.sort()
    y_points_split.sort()

    print(x_points_split, y_points_split)
    obtained_cells = split_cells(x_points_split, y_points_split)
    return obtained_cells

  def find_chess_board_cells(img, debug_mode):

    if debug_mode:
      cv2.imshow('input image',img)    
      cv2.waitKey()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray, 7) # 9

    # Make binary image
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 9,9) #10 for big #6 for small
    

    if debug_mode:
      cv2.imshow('binary image',bin_img)    
      cv2.waitKey()


    # Detect lines
    rho = 2
    theta = np.pi/180 * 0.5
    thresh = 300   #400
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)

    if sys.version_info[0] == 2:
      # python 2.x
      # Re-shape from 1xNx2 to Nx1x2
      temp_lines = []
      N = lines.shape[1]
      for i in range(N):
        rho = lines[0,i,0]
        theta = lines[0,i,1]
        temp_lines.append( np.array([[rho,theta]]) )
      lines = temp_lines

    print("Found lines: %d" % (len(lines)))


    # Cluster line angles into 2 groups (vertical and horizontal)
    segmented = segment_by_angle_kmeans(lines, 2)

    # Find the intersections of each vertical line with each horizontal line
    intersections = segmented_intersections(segmented)


    # display hough lines detected
    if debug_mode:    
      img_copy = np.copy(img)

      # Draw vertical lines in green
      vertical_lines = segmented[0]
      drawLines(img_copy, vertical_lines, (0,255,0))

      # Draw horizontal lines in yellow
      horizontal_lines = segmented[1]
      drawLines(img_copy, horizontal_lines, (0,255,255))

      cv2.imshow('hough lines detected',img_copy)
      cv2.waitKey()
    
    intersections = segmented_intersections(segmented)

    # display intersection points
    if debug_mode:
      img_copy = np.copy(img)    

      for point in intersections:
        pt = (point[0][0], point[0][1])
        length = 5
        cv2.line(img_copy, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 2) # vertical line
        cv2.line(img_copy, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 2)


      cv2.imshow('intersection points',img_copy)
      cv2.waitKey()
    
    # finding groups

    flag_array = np.zeros((len(intersections)), dtype=int) # array to store group number

    flag_number = 1

    # sorting points

    # first loop

    for idx in range(len(intersections)):

      if flag_array[idx] == 0:

        first_point = intersections[idx]

        first_pt = (first_point[0][0], first_point[0][1])

        flag_array[idx] = flag_number

        flag_number = flag_number + 1

        # second loop

        for idx1 in range(len(intersections)):
          
          if flag_array[idx1] == 0:

            point = intersections[idx1]

            pt = (point[0][0], point[0][1])

            dist_ = distance.euclidean(first_pt, pt) # finding distance

            if dist_ < 20: # given threshold to 50

              flag_array[idx1] = flag_number - 1


    new_intersection = [] 

    for group in range(max(flag_array)):  # iterate over number of groups
      x_ = 0
      y_ = 0
      n_ = 0

      for idx in range(len(intersections)):

        if flag_array[idx] == group + 1:

          x_ = x_ + intersections[idx][0][0]
          y_ = y_ + intersections[idx][0][1]
          n_ = n_ + 1

      x_ = int(x_/n_) # finding mean
      y_ = int(y_/n_)

      new_intersection.append([[x_,y_]])


    #print(intersections[0:20] ,'\n', flag_array[0:20])

    #print(len(new_intersection))

    if debug_mode:

      img_copy = np.copy(img)

      for point in new_intersection:
        pt = (point[0][0], point[0][1])
        length = 5
        cv2.line(img_copy, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 5) # vertical line
        cv2.line(img_copy, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 5)
        text = str(pt[0])+','+ str(pt[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_copy,text,(pt[0],pt[1]), font, .45,(255,100,0),1,cv2.LINE_AA)

      cv2.imshow('filtered intersection points',img_copy)
      cv2.waitKey()

    
    # rearanging array

    newinter = []

    for single in new_intersection:
      newinter.append((round(single[0][0],-1), round(single[0][1],-1)))


    # finding distance according to x and y axis

    sorted_wrt_y = sorted(newinter , key=lambda k: [k[0], k[1]])
    sorted_wrt_x = sorted(newinter , key=lambda k: [k[1], k[0]])

    # finding distances along x axis

    temp_dist = []
    for idx in range((len(sorted_wrt_x)-1)):
      point1 = (sorted_wrt_x[idx][0],sorted_wrt_x[idx][1])
      point2 = (sorted_wrt_x[idx+1][0],sorted_wrt_x[idx+1][1])
      temp_dist.append(distance.euclidean(point1, point2))

    # finding most repeated distance of x axis using histogram
    temp_dist = [round(x, -1) for x in temp_dist]
    n_x, b_x, patches = plt.hist(temp_dist, bins = 500) #500
    bin_max_x = np.where(n_x == n_x.max())
    x_max = int(np.mean(b_x[bin_max_x]))
    x_range = range(x_max - int(0.1*x_max)  , x_max + int(0.1*x_max)) # setting range for points for selection .1, .5

    #print(x_max)

    # finding distances along y axis

    temp_dist = []
    for idx in range((len(sorted_wrt_y)-1)):
      point1 = (sorted_wrt_y[idx][0],sorted_wrt_y[idx][1])
      point2 = (sorted_wrt_y[idx+1][0],sorted_wrt_y[idx+1][1])
      temp_dist.append(distance.euclidean(point1, point2))

    # finding most repeated distance of y axis using histogram
    temp_dist = [round(x, -1) for x in temp_dist]
    n_y, b_y, patches = plt.hist(temp_dist, bins = 300)
    bin_max_y = np.where(n_y == n_y.max())
    y_max = int(np.mean(b_y[bin_max_y]))
    y_range = range(y_max - int(0.1*y_max)  , y_max + int(0.1*y_max)) # setting range for points for selection .1,.1


    #print(y_max)
    # sorting according to distance

    newinter_sorted_x = []
    newinter_sorted_y = []


    # selecting points through x axis
    temp_dist_ = 0

    for idx in range((len(sorted_wrt_x)-1)):
      point1 = (sorted_wrt_x[idx][0],sorted_wrt_x[idx][1])
      point2 = (sorted_wrt_x[idx+1][0],sorted_wrt_x[idx+1][1])
      temp_dist_ = int(distance.euclidean(point1, point2))
      if temp_dist_ in x_range:
        newinter_sorted_x.append(point1) 

    # selecting points through y axis
    temp_dist_ = 0

    for idx in range((len(sorted_wrt_y)-1)):
      point1 = (sorted_wrt_y[idx][0],sorted_wrt_y[idx][1])
      point2 = (sorted_wrt_y[idx+1][0],sorted_wrt_y[idx+1][1])
      temp_dist_ = int(distance.euclidean(point1, point2))
      if temp_dist_ in y_range:
        newinter_sorted_y.append(point1) 
        


    #sorting in reverse direction
    sorted_wrt_y_ = sorted(newinter , key=lambda k: [k[0], k[1]] , reverse=True)
    sorted_wrt_x_ = sorted(newinter , key=lambda k: [k[1], k[0]] , reverse=True)


    # sorting according to distance

    newinter_sorted_x_ = []
    newinter_sorted_y_ = []


    # selecting points through reverse x axis
    temp_dist_ = 0

    for idx in range((len(sorted_wrt_x_)-1)):
      point1 = (sorted_wrt_x_[idx][0],sorted_wrt_x_[idx][1])
      point2 = (sorted_wrt_x_[idx+1][0],sorted_wrt_x_[idx+1][1])
      temp_dist_ = int(distance.euclidean(point1, point2))
      if temp_dist_ in x_range:
        newinter_sorted_x_.append(point1) 


    # selecting points through reverse y axis
    temp_dist_ = 0

    for idx in range((len(sorted_wrt_y_)-1)):
      point1 = (sorted_wrt_y_[idx][0],sorted_wrt_y_[idx][1])
      point2 = (sorted_wrt_y_[idx+1][0],sorted_wrt_y_[idx+1][1])
      temp_dist_ = int(distance.euclidean(point1, point2))
      if temp_dist_ in y_range:
        newinter_sorted_y_.append(point1) 




    new_inter_x = set(newinter_sorted_x + newinter_sorted_x_)
    #new_inter_x = newinter_sorted_x
    #new_inter_x = list(set(newinter_sorted_x).intersection(newinter_sorted_x_))
    new_inter_y = set(newinter_sorted_y + newinter_sorted_y_)
    #new_inter_y = newinter_sorted_y
    #new_inter_y = list(set(newinter_sorted_y).intersection(newinter_sorted_y_))
    new_inter_final = list(set(new_inter_x).intersection(new_inter_y))
    #new_inter_final = set(new_inter_x + new_inter_y)



    new_intersection_test = []
    x_points_obtained = []
    y_points_obtained = []

    resultpoints = 'xrevx'

    if resultpoints == 'full':
      for pt in new_inter_final:
        new_intersection_test.append([[pt[0], pt[1]]])
        x_points_obtained.append(pt[0])
        y_points_obtained.append(pt[1])

    elif resultpoints == 'xrevx':
      for pt in new_inter_x:
        new_intersection_test.append([[pt[0], pt[1]]])
        x_points_obtained.append(pt[0])
        y_points_obtained.append(pt[1])

    elif resultpoints == 'yrevy':
      for pt in new_inter_y:
        new_intersection_test.append([[pt[0], pt[1]]])
        x_points_obtained.append(pt[0])
        y_points_obtained.append(pt[1])

    elif resultpoints == 'x':
      for pt in newinter_sorted_x:
        new_intersection_test.append([[pt[0], pt[1]]])
        x_points_obtained.append(pt[0])
        y_points_obtained.append(pt[1])

    elif resultpoints == 'revx':
      for pt in newinter_sorted_x_:
        new_intersection_test.append([[pt[0], pt[1]]])
        x_points_obtained.append(pt[0])
        y_points_obtained.append(pt[1])

    elif resultpoints == 'y':
      for pt in newinter_sorted_y:
        new_intersection_test.append([[pt[0], pt[1]]])
        x_points_obtained.append(pt[0])
        y_points_obtained.append(pt[1])

    elif resultpoints == 'revy':
      for pt in newinter_sorted_y_:
        new_intersection_test.append([[pt[0], pt[1]]])
        x_points_obtained.append(pt[0])
        y_points_obtained.append(pt[1])     



    if debug_mode:
      img_copy = np.copy(img)

      for point in new_intersection_test:
        pt = (point[0][0], point[0][1])
        length = 5
        cv2.line(img_copy, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 5) # vertical line
        cv2.line(img_copy, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 5)

      cv2.imshow(resultpoints,img_copy)
      cv2.waitKey()
    
    

    values_x_ = []
    values_y_ = []

    #data = x_points_obtained

    maxgap = 10

    values_x_ = grouping_and_mean(x_points_obtained, maxgap)
    values_y_ = grouping_and_mean(y_points_obtained, maxgap)

    #print('before mean',values_x_ , values_y_)


    # adding codes
    mean_x = np.mean(values_x_)
    dist_map = []
    for item in values_x_:
      dist_temp = abs(mean_x-item)
      dist_map.append([dist_temp,item])

    dist_map=np.array(dist_map)
    dist_map = dist_map[np.lexsort(dist_map.T[::-1])]

    #print(dist_map)

    values_x_ = []
    for item in range(9):
      values_x_.append(int(dist_map[item][1]))

    values_x_.sort()

    mean_y = np.mean(values_y_)
    dist_map = []
    for item in values_y_:
      dist_temp = abs(mean_y-item)
      dist_map.append([dist_temp,item])

    dist_map=np.array(dist_map)
    dist_map = dist_map[np.lexsort(dist_map.T[::-1])]

    values_y_ = []
    for item in range(9):
      values_y_.append(int(dist_map[item][1]))

    values_y_.sort()

    #print('after mean',values_x_ , values_y_)

    new_intersection_final = []

    # check weather points exit or not

    for i in range(len(values_x_)):
      for j in range(len(values_y_)):

        range_x_new = range(values_x_[i]-25, values_x_[i]+25)
        range_y_new = range(values_y_[j]-25, values_y_[j]+25)
        
        for a in new_intersection:
          if a[0][0] in range_x_new and a[0][1] in range_y_new:
            new_intersection_final.append(a)
            break
          
    
    
    if debug_mode:
      img_copy = np.copy(img)

      for point in new_intersection_final:
        pt = (point[0][0], point[0][1])
        length = 5
        cv2.line(img_copy, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 5) # vertical line
        cv2.line(img_copy, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 5)

      cv2.imshow('final detection',img_copy)
      cv2.waitKey()

      cv2.destroyAllWindows()

    return new_intersection_final

  def extract_xy(chess_points):
    x_points_obtained = []
    y_points_obtained = []

    for pt in chess_points:    
      x_points_obtained.append(pt[0][0])
      y_points_obtained.append(pt[0][1])
      
    return x_points_obtained, y_points_obtained
