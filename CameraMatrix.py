import cv2
import numpy as np
import os
import glob
import pandas as pd
import re


class CameraMatrix:

    def __init__(self):
        self.CHECKERBOARD = (6, 9)

        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Vector for 3D points
        self.threedpoints = []

    # Vector for 2D points
        self.twodpoints = []

    #  3D points real world coordinates
        self.objectp3d = np.zeros((1, self.CHECKERBOARD[0]
                                   * self.CHECKERBOARD[1],
                                   3), np.float32)
        self.objectp3d[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0],
                                            0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        self.prev_img_shape = None

    def InputImage(self, filename):
        # Extracting path of individual image stored
        # in a given directory. Since no path is
        # specified, it will take current directory
        # jpg files alone
        self.images = glob.glob('{}'.format(filename))
        print('images', self.images)

    def ComputeCamMat(self):

        for filename in self.images:
            image = cv2.imread(filename)
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            # If desired number of corners are
            # found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(
                grayColor, self.CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK +
                cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:
                self.threedpoints.append(self.objectp3d)

                # Refining pixel coordinates
                # for given 2d points.
                corners2 = cv2.cornerSubPix(
                    grayColor, corners, (11, 11), (-1, -1), self.criteria)

                self.twodpoints.append(corners2)

                # Draw and display the corners
                image = cv2.drawChessboardCorners(image,
                                                  self.CHECKERBOARD,
                                                  corners2, ret)

            cv2.imshow('img', image)
            # cv2.waitKey(0)

        cv2.destroyAllWindows()

        h, w = image.shape[:2]


# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
            self.threedpoints, self.twodpoints, grayColor.shape[::-1], None, None)

        # Displaying required output
        print(" Camera matrix:")
        print(matrix)

        print("\n Distortion coefficient:")
        print(distortion)

        print("\n Rotation Vectors:")
        print(r_vecs)

        print("\n Translation Vectors:")
        print(t_vecs)

        return matrix, distortion, r_vecs, t_vecs

    def concatMat(self, matrix, r_vecs, t_vecs):
        zero_mat = np.array([[0, 0, 0]])
        print('zero_mat', zero_mat)
        inter = np.concatenate((matrix, zero_mat), axis=0)
        print('inter', inter)

        # shape of the tranlation vector
        #print('np.shape(t_vecs)', np.shape(t_vecs))
        #print('np.shape(t_vecs)', np.shape(t_vecs)[0])

        # shape of the rotation vector
        #print('np.shape(r_vecs)', np.shape(r_vecs))
        #np.concatenate((inter, ))

    def obtain_PT(self, filenames):
        # filenames=glob.glob('{}'.format(filenames))

        # print(self.filenames)
        self.filenames = filenames

        regex = re.compile(r'\d+')
        regex.findall(self.filenames)
        s = [int(x) for x in regex.findall(self.filenames)]

        pan = s[1]
        tilt = s[0]

        print('pan', pan)
        print('tilt', tilt)
        return pan, tilt

    def read_csvfile(self):

        self.filenames = glob.glob('{}'.format(self.filenames))
        for i in self.filenames:

            print('self.filename', i)
            self.data = pd.read_csv(i, delimiter=',')
            print('self.data', self.data)
            print('type of self.data', type(self.data))
            #self.data['x'] =self.data['x'].map(lambda x: x.split(' ').lstrip('[[   ').rstrip(']]'))
            print('self.data', self.data)
            #self.data=np.loadtxt(self.filename, dtype=float)
            # with open(i, 'r') as file:
            #     self.data=file.read()
            # print('self.data', self.data)
            # self.data=pd.DataFrame(self.data)
            #print('self.data pd form', self.data)
            #print('type of self.data \n \n', type(self.data))
            self.data = np.array(self.data)
            # print('self.data', self.data)
            #print('self.data pd form', self.data)
            #print('type of self.data \n \n', type(self.data))
            # print(np.shape(self.data))

            # self.data=np.array([self.data])
            # print('self.data \n \n', self.data)
            # print('type of self.data \n \n', type(self.data))
            # print('counter',counter)

            x = self.data[:, 0]
            y = self.data[:, 1]

            # print('x', x)
            # print('y', y)
            return x, y

    def compute_deltaPanTilt(self, pan0, pan1, tilt0, tilt1):
        self.delta_pan = pan1-pan0
        #print('delta_pan', self.delta_pan)

        self.delta_tilt = tilt1-tilt0
        #print('delta_tilt', self.delta_tilt)

        self.delta_pan = np.repeat(self.delta_pan, 53, axis=0)
        #print('delta_pan', delta_pan)
        self.delta_pan = np.reshape(self.delta_pan, (53, 1))

        self.delta_tilt = np.repeat(self.delta_tilt, 53, axis=0)
        #print('delta_pan', delta_pan)
        self.delta_tilt = np.reshape(self.delta_tilt, (53, 1))
        print('delta_pan', self.delta_pan)
        print('delta_tilt', self.delta_tilt)

    def compute_matA(self, x, x1, y, y1):
        delta_x = np.subtract(x1, x)
        delta_y = np.subtract(y1, y)

        # print(np.shape(delta_x))
        # print(np.shape(delta_y))
        delta_x = np.reshape(delta_x, (53, 1))
        delta_y = np.reshape(delta_y, (53, 1))
        #print('delta_x', delta_x)
        #print('delta_y', delta_y)

    # concatenate delta x and y to create the matrix a

        A = np.concatenate((delta_x, delta_y), axis=1)
        #print('A', A)
        #print('shape A', np.shape(A))

        A_inv = np.linalg.pinv(A)
        #print('A_inv', A_inv)

        #print('np.shape(A_inv)', np.shape(A_inv))
        ab = np.dot(A_inv, self.delta_pan)
        # ab=abs(ab)
        print('ab', ab)

        cd = np.dot(A_inv, self.delta_tilt)
        # cd=abs(cd)
        print('cd', cd)
        return ab, cd


if __name__ == '__main__':
    camat = CameraMatrix()

    # pan0, tilt0 = camat.obtain_PT('*b_corners_updown623_right_tilt586.csv')
    # x0, y0 = camat.read_csvfile()

    # pan1, tilt1 = camat.obtain_PT('*b_corners_updown633_right_tilt586.csv')
    # x1, y1 = camat.read_csvfile()

    # pan2, tilt2 = camat.obtain_PT('*b_corners_updown666_right_tilt586.csv')
    # x2, y2 = camat.read_csvfile()

    # pan3, tilt3 = camat.obtain_PT('*b_corners_updown694_right_tilt586.csv')
    # x3, y3 = camat.read_csvfile()

    # # compute delta pan and delta tilt

    # delta_pan1 = pan1-pan0
    # delta_pan2 = pan2-pan1
    # delta_pan3 = pan3-pan2

    # delta_tilt1 = tilt1-tilt0
    # delta_tilt2 = tilt2-tilt1
    # delta_tilt3 = tilt3-tilt2

    # delta_motor = np.array([[delta_pan1, delta_pan2, delta_pan3],  [
    #                        delta_tilt1, delta_tilt2, delta_tilt3]])

    # delta_x_1 = x1-x0
    # delta_x_2 = x2-x1
    # delta_x_3 = x3-x2
    # # print(delta_x_1)

    # delta_y_1 = y1-y0
    # delta_y_2 = y2-y1
    # delta_y_3 = y3-y2

    # #print('delta_x_1', delta_x_1)

    # # print('shape of delta_x_1', np.shape(delta_x_1)) # 53 rows
    # delta_x_1 = np.reshape(delta_x_1, (53, -1))
    # delta_x_2 = np.reshape(delta_x_2, (53, -1))
    # delta_x_3 = np.reshape(delta_x_3, (53, -1))

    # delta_y_1 = np.reshape(delta_y_1, (53, -1))
    # delta_y_2 = np.reshape(delta_y_2, (53, -1))
    # delta_y_3 = np.reshape(delta_y_3, (53, -1))

    # # print('delta_x_1', delta_x_1)
    # # print('shape of delta_x_1', np.shape(delta_x_1)) # 53 rows

    # # for i in np.nditer(delta_x_1):
    # #     print(i, end=' ')

    # delta_x = np.hstack([delta_x_1, delta_x_2, delta_x_3])
    # #print('delta_x', delta_x)
    # #print('sjape of deltax', np.shape(delta_x))

    # delta_y = np.hstack([delta_y_1, delta_y_2, delta_y_3])

    # delta_image = []

    # for i in range(53):

    #     s = np.vstack([delta_x[i], delta_y[i]])
    #     #print('xy', s)
    #     #print('shape of xy', np.shape(s))
    #     delta_image.append(s)

    # #print('delta_image', delta_image)
    # #print('shape of delta image', np.shape(delta_image))
    # A_matrix = []
    # Z = 0
    # B = 0
    # C = 0
    # D = 0

    # for i in range(53):
    #     #print('image matrix {}:  '.format(i),delta_image[i])
    #     t = np.dot(delta_image[i], delta_image[i].T)
    #     t = np.linalg.inv(t)
    #     t = np.dot(delta_image[i].T, t)

    #     A = np.dot(delta_motor, t)
    #     #print('{}th A: \n '.format(i), A)
    #     A_matrix.append(A)
    #     # calculating average of C
    #     #A[1, 0]=0
    #     Z += A[0, 0]
    #     B += A[0, 1]

    #     C += A[1, 0]
    #     D += A[1, 1]
    #     #print('C', C)
    #     #print(A[0,0], A[0,1], A[1, 0], A[1, 1])
    # Avg_A = Z/53
    # Avg_B = B/53
    # Avg_C = C/53
    # Avg_D = D/53
    # print('A_matrix: \n  ', A_matrix)

    # print('Avg_C', Avg_C)
    # print('Avg_D', Avg_D)

# 1
x = 46.88335
y = 102.65005

# 10

x10 = 66.10497
y10 = 51.619347

x20 = 100.560295
y20 = 88.48509

x30 = 118.61483
y30 = 22.492905

x40 = 149.52356
y40 = 59.179184

dx=x10-x
dx1=x20-x10
dx2=x30-x20
dx3=x40-x30

dy=y10-y
dy1=y20-y10
dy2=y30-y20
dy3=y40-y30
