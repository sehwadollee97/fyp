# https://learnopencv.com/camera-calibration-using-opencv/
# https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/

from turtle import left
import cv2
import yaml.dumper
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sophia import Sophia
from AECControl import AECControl

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


# Extracting path of individual image stored in a given directory
# dir=os.getcwd()
# print('dir', dir)
# images = cv2.imread('dir{}/left350.png'.format(dir))
left = Sophia(14)
right = Sophia(15)
updown = Sophia(16)

# _, _, mid_r=right.get_limit()
# right.move(mid_r)

# _, _, mid_u=updown.get_limit()
# updown.move(mid_u)

# _, _, mid_l=left.get_limit()
# left.move(mid_l)


# ####################################################################################
# #calibration

# min_left = left.calibration(0)
# max_left = left.calibration(1)
# left.move((min_left+max_left)/2)
#left.write()



# min_right = right.calibration(0)
# max_right = right.calibration(1)
# right.move((min_right+max_right)/2)
#right.write()


# min_updown = updown.calibration(0)
# max_updown = updown.calibration(1)
# updown.move((min_updown+max_updown)/2)
#updown.write()

# print('min left: {0} & max left:{1}  '.format(min_left,max_left) )
# # print('min right: {0} & max right: {1}'.format( min_right,max_right))
# print('min updown: {0}, max_updown: {1}'.format(min_updown,max_updown))

# #obain the current position of the left and right
# left_currentpos=left.get_currentPos()
# right_currentpos=right.get_currentPos()
# updown_currentpos=updown.get_currentPos()

#print('current left: {0}, current right: {1}, current updown: {2}'.format( left_currentpos, right_currentpos, updown_currentpos))
####################################
position_left_prev=left.get_currentPos()
position_right_prev=right.get_currentPos()
position_updown_prev=updown.get_currentPos()

print('position_right_prev', position_right_prev)
print('position_updown_prev', position_updown_prev)
print('position_left_prev', position_left_prev)
#movement
right.moveby(-10)
#updown.moveby(10)


position_right=right.get_currentPos()
position_updown=updown.get_currentPos()
position_left=left.get_currentPos()

#position=updown.get_currentPos()
print('position_right', position_right)
print('position_updown', position_updown)
print('position_left', position_left)


aec=AECControl()
left_img, right_img=aec.capture() 
plt.figure()
plt.imshow(right_img)
plt.show()


#images=glob.glob('./*L?.png')
for fname in left_img:
    img=right_img
    
    
    #img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    print('ret:  ', ret)
    if ret == True:
        mpimg.imsave('b_updown{}_right{}_pan.png'.format(position_updown, position_right),right_img)
        #plt.savefig('right_img.png')
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # writing the corners to the txt file
        #print('range(0, len(right_img))',  len(right_img))
        

        with open('b_corners_updown{}_right_pan{}.csv'.format(position_updown, position_right), 'w') as f:
            for line in corners2:
                line=str(line)
                f.writelines(line) # writing tuples
                f.writelines('\n') # writing tuples


        print('corners:  ', corners2)
        print('type of corners:  ', type(corners2))
        print('size of corners:  ', (np.shape(corners2)))
        #print('corner2 {0}th row, 1st column, {1}th element'.format(1,1), corners2[0][0][0])
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img_updown{}_right{}'.format(position_updown, position_right),img)
        
        file_dir='/home/fyp/Downloads/SCServo_Python_200821/right eye calibration'
        
        mpimg.imsave('b_img_updown{}_right{}_pan.png'.format(position_updown, position_right), img)
    
    #cv2.imshow('img',img)
    cv2.waitKey(0)

	 
	
        


        

# 		#fname = "/home/fyp/Downloads/SCServo_Python_200831/SCServo_Python/feetechsmall.yaml"
		

#     #cv2.imshow('img', img)
#     #cv2.waitKey(0)

# cv2.destroyAllWindows()

# h, w = img.shape[:2]

# """
# Performing camera calibration by 
# passing the value of known 3D points (objpoints)
# and corresponding pixel coordinates of the 
# detected corners (imgpoints)
# """
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1], None, None)

# print("Camera matrix : \n")
# print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)

















