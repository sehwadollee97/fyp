import pickle
import time
import os
import math
import numpy as np
import cv2
import random

import scipy.io as scio
import matplotlib.pyplot as plt

# environment
from numpy.lib.shape_base import expand_dims
from skimage.util import view_as_windows

from AECControl import AECControl


class Agent:

    def __init__(self):
        self.interval = 5  # The period to change a new environment for the eye
        self.saccadenum = 3  # The number of saccade actions in one scene
        self.timeTrain = int(2e5)  # Training iterations

        self.saccmd = [0, 0]
        self.vergcmd = 0
        # self.imagedata = imageData

        self.coarseScSize = [220, 220]  # size of image
        self.halfSizeCoarseSc = [
            int(self.coarseScSize[0]/2), int(self.coarseScSize[1]/2)]

        self.curImSize = [480, 640]  # CurrentImage size
        self.saccadeAngle = 0
        # self.randWalkDist = np.zeros((1,2),dtype=int)

        picidx = 0  # picture index
        self.interCounter = 0
        self.interC2 = 0

        #######
        self.saccmd_x = 0
        self.saccmd_y = 0

        self.eye_gaze_x_init = 0
        self.eye_gaze_y_init = 0
        # model = [self.imagedata, self.w_filter, self.initialbases,
        #         self.transProb, self.initialweights]
        # self.imagedata = PARAM["imagedata"]

    def ImageProcessing(self, imagedata, patch_size, fx, fy, robotype):
        self.patch_size = patch_size
        self.imagedata = imagedata
        self.fx = fx
        self.fy=fy
        self.robotype = robotype

        if robotype == 1:

            data = list(self.imagedata.items())
            self.imagedata = np.array(data)
            self.processed_imagedata = self.imagedata[3, 1]
            self.img_dim_y = np.size(self.processed_imagedata[1, 1], 1)
            self.img_dim_x = np.size(self.processed_imagedata[1, 1], 0)

            # compute patch size
            self.w = self.patch_size/2
            self.w = int(self.w)
            print('w:   ', self.w)

        else:
            self.img_dim_y = np.size(self.imagedata, 0)
            self.img_dim_x = np.size(self.imagedata, 1)

            # compute patch size
            self.w = self.patch_size/2
            self.w = int(self.w)
            print('w:   ', self.w)

        # print('imagedata[1, 1]', processed_imagedata[1, 1])
        # print('size of processed_imagedata[1, 1]', np.size(
        #     processed_imagedata[1, 1], 0))
        # print('size of processed_imagedata[1, 1]', np.size(
        #     processed_imagedata[1, 1], 1))
        return self.img_dim_x, self.img_dim_y

    def generate_randomTargetCoord(self):

        eye_gaze_x = 0
        eye_gaze_y = 0

        # saccade at the centre
        saccmd_x = int(self.img_dim_x/2)
        saccmd_y = int(self.img_dim_y/2)

        self.t_x = random.randint(-80, 80)
        self.t_y = random.randint(-80, 80)

        print('t_x:   ', self.t_x)
        print('t_y:   ', self.t_y)

    def generate_eyeCmd(self, pan_left_scale, pan_right_scale, tilt_left_scale, tilt_right_scale):

        self.pan_left_scale=pan_left_scale
        self.pan_right_scale=pan_right_scale
        self.tilt_left_scale=tilt_left_scale
        self.tilt_right_scale=tilt_right_scale
        
        
        # ensure the limit of the x and y positions to be within the image +-w
        if (self.t_x <= self.w) or (self.t_x >= self.img_dim_x-self.w):
            self.t_x = random.randint(self.w+1,  self.img_dim_x-self.w-1)
        if self.t_y <= self.w or self.t_y >= self.img_dim_y-self.w:
            self.t_y = random.randint(self.w+1, self.img_dim_y-self.w-1)

        print('t_x after boundary:   ', self.t_x)
        print('t_y after boundary:   ', self.t_y)
        self.x_before=self.t_x-self.w
        self.x_after=self.t_x+self.w

        self.y_before=self.t_y-self.w
        self.y_after=self.t_y+self.w

        self.im_BS = self.imagedata[self.y_before:self.y_after, self.x_before:self.x_after]
        # set focal length

        print('self.im_BS x:   ', np.size(self.im_BS, axis=0))
        print('self.im_BS y:   ', np.size(self.im_BS, axis=1))
        

        print('coordinates of im_BS (x-w, x, x+w, y-w, y,  y+w)',
                (self.t_x-self.w, self.t_x, self.t_x+self.w, self.t_y-self.w, self.t_y, self.t_y+self.w))

        # compute the the saccade command for x and y coordinates


        pan = ((self.img_dim_x/2-self.t_x)/self.pan_left_scale) # how much to move horizontally in motor units
        tilt = ((self.img_dim_y/2-self.t_y)/self.tilt_left_scale) # how much to move vertically in motor units
        # update the eye gaze x and y coordinates
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('original image with target')
        plt.imshow(self.imagedata)
        plt.scatter(self.t_x, self.t_y, color='red')

        plt.title('BS')

        plt.subplot(1, 2, 2)
        plt.imshow(self.im_BS)

        plt.show()
        
        #                 marker='o')  # target coordinates
        return pan, tilt, self.t_x, self.t_y

    def compute_saccade_positions(self, pan_valid, tilt_valid):

        # if self.robotype == 1:

        #     self.eye_gaze_x = self.f*math.tan(pan_valid)
        #     self.eye_gaze_y = self.f*math.tan(tilt_valid)
        #     print('eye_gaze_x:  ', self.eye_gaze_x)
        #     print('eye_gaze_y:  ', self.eye_gaze_y)

        #     self.eye_gaze_x = int(round(self.eye_gaze_x, 0))
        #     self.eye_gaze_y = int(round(self.eye_gaze_y, 0))

        #     print('rounded eye_gaze_x:  ', self.eye_gaze_x)
        #     print('rounded eye_gaze_y:  ', self.eye_gaze_y)

        #     self.im_BS = self.processed_imagedata[1, 1][(
        #         self.t_x-self.w):(self.t_x+self.w), (self.t_y-self.w):(self.t_y+self.w)]
        #     self.im_AS = self.processed_imagedata[1, 1][(
        #         self.eye_gaze_x-self.w):(self.eye_gaze_x+self.w), (self.eye_gaze_y-self.w):(self.eye_gaze_y+self.w)]

        #     # compute correlation
        #     corr = self.compute_corr_coef()

        #     print('self.im_BS x:   ', np.size(self.im_BS, axis=0))
        #     print('self.im_BS y:   ', np.size(self.im_BS, axis=1))
        #     print('self.im_AS x:   ', np.size(self.im_AS, axis=0))
        #     print('self.im_AS y:   ', np.size(self.im_AS, axis=1))

        #     print('coordinates of im_BS (x-w, x, x+w, y-w, y,  y+w)',
        #           (self.t_x-self.w, self.t_x, self.t_x+self.w, self.t_y-self.w, self.t_y, self.t_y+self.w))
        #     print('coordinates of im_AS (x-w,x, x+w, y-w,y, y+w)', (self.eye_gaze_x-self.w,
        #                                                             self.eye_gaze_x, self.eye_gaze_x+self.w, self.eye_gaze_y-self.w, self.eye_gaze_y, self.eye_gaze_y+self.w))

        #     plt.figure(figsize=(8, 8))
        #     plt.subplot(2, 2, 1)
        #     plt.scatter(self.eye_gaze_x, self.eye_gaze_y, color='White',
        #                 marker='X')  # after saccade

        #     plt.scatter(self.t_x, self.t_y, color='red',
        #                 marker='o')  # target coordinates
        #     # plt.scatter(new_eye_x, new_eye_y, color='blue', marker='*')
        #     plt.title('Original image')
        #     plt.imshow(self.processed_imagedata[1, 1])

        #     plt.subplot(2, 2, 3)
        #     plt.imshow(self.im_BS)
        #     plt.title('Before Saccade')

        #     plt.subplot(2, 2, 4)
        #     plt.imshow(self.im_AS)
        #     plt.title('After Saccade')
        #     plt.show()

        # else:
            # self.eye_gaze_x = self.fx*math.tan(pan_valid)
            # self.eye_gaze_y = self.fy*math.tan(tilt_valid)
            # print('eye_gaze_x:  ', self.eye_gaze_x)
            # print('eye_gaze_y:  ', self.eye_gaze_y)

            # self.eye_gaze_x = int(round(self.eye_gaze_x, 0))
            # self.eye_gaze_y = int(round(self.eye_gaze_y, 0))

            # print('rounded eye_gaze_x:  ', self.eye_gaze_x)
            # print('rounded eye_gaze_y:  ', self.eye_gaze_y)

        # after saccade
        # self.eye_gaze_x = self.fx*math.tan(pan_valid)
        # self.eye_gaze_y = self.fy*math.tan(tilt_valid)
        # print('eye_gaze_x:  ', self.eye_gaze_x)
        # print('eye_gaze_y:  ', self.eye_gaze_y)

        # self.eye_gaze_x = int(round(self.eye_gaze_x, 0))
        # self.eye_gaze_y = int(round(self.eye_gaze_y, 0))

        # print('rounded eye_gaze_x:  ', self.eye_gaze_x)
        # print('rounded eye_gaze_y:  ', self.eye_gaze_y)


        #computing the direction of the pan and tilt

        print('self.img_dim_x:  ', self.img_dim_x)
        print('self.img_dim_y:  ', self.img_dim_y)

        print('self.tx:  ', self.t_x)
        print('self.ty:  ', self.t_y)
       
        if self.t_x < self.img_dim_x/2:
            print('tx is less than the img_dim_x, so move to the left')
            pan_valid=-pan_valid

        elif self.t_x > self.img_dim_x/2:
            print('tx is greater than the img_dim_x, so move to the right')
            pan_valid=pan_valid
        else:
            print('tx is equal to the img_dim_x')
            pan_valid=0
        

        print('directional pan_valid:  ', pan_valid)

        if self.t_y > (self.img_dim_y/2):
            print('ty is greater than the img_dim_y, so move down')
            tilt_valid=-tilt_valid

        elif self.t_y < (self.img_dim_y/2):
            print('ty is less than than the img_dim_y, so move up')
            tilt_valid=tilt_valid
        else:
            print('ty is equal to the img_dim_y')
            tilt_valid=0
        
        print('directional tilt_valid:  ', tilt_valid)
        
        return pan_valid, tilt_valid





    def plot_images(self,AS_IMG):
        self.AS_IMG=AS_IMG


        self.im_BS = self.imagedata[self.y_before:self.y_after, self.x_before:self.x_after]
        # set focal length

        self.center_x=self.img_dim_x/2
        self.center_y=self.img_dim_y/2


        print('self.im_BS x:   ', np.size(self.im_BS, axis=0))
        print('self.im_BS y:   ', np.size(self.im_BS, axis=1))
        self.im_AS = self.AS_IMG[int(self.center_x -self.w):int(self.center_x+self.w), int(self.center_y -self.w):int(self.center_y +self.w)]

        corr = self.compute_corr_coef()

        print('self.im_BS x:   ', np.size(self.im_BS, axis=0))
        print('self.im_BS y:   ', np.size(self.im_BS, axis=1))
        print('self.im_AS x:   ', np.size(self.im_AS, axis=0))
        print('self.im_AS y:   ', np.size(self.im_AS, axis=1))

        print('coordinates of im_BS (x-w, x, x+w, y-w, y,  y+w)',
                (self.t_x-self.w, self.t_x, self.t_x+self.w, self.t_y-self.w, self.t_y, self.t_y+self.w))
        print('coordinates of im_AS (x-w,x, x+w, y-w,y, y+w)', 
        int(self.center_y -self.w),int(self.center_x+self.w), int(self.center_x -self.w),int(self.center_x +self.w))

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        #plt.scatter(self.eye_gaze_x, self.eye_gaze_y, color='White',
        #            marker='X')  # after saccade

        plt.scatter(self.t_x, self.t_y, color='red',
                    marker='o')  # target coordinate
        plt.title('Original image')
        plt.imshow(self.imagedata)

        plt.subplot(2, 2, 3)
        plt.imshow(self.im_BS)
        plt.title('Before Saccade')

        plt.subplot(2, 2, 4)
        plt.imshow(self.im_AS)
        plt.title('After Saccade')
        plt.show()



        # plt.subplot(2, 2, 4)
        # plt.imshow(self.im_AS)
        # plt.title('After Saccade')
        # plt.show()

        # resize im_BS and im_AS to get corrcoef

    
# ------------------------------------------------------------------------------------------

    def compute_corr_coef(self):
        #print('self.im_BS', self.im_BS)
        #print('shape of im_BS', np.shape(self.im_BS))
        self.im_BS_resize = np.resize(self.im_BS, (1, self.patch_size))

        self.im_AS_resize = np.resize(self.im_AS, (1, self.patch_size))

        corr = np.corrcoef(self.im_BS_resize, self.im_AS_resize)
        print('Correlation:   ', np.fliplr(corr).diagonal())

        return corr

        # compute correlation coefficients between imL and imR

        ######################################################################################################
        ######################################################################################################
        ######################################################################################################


def getfixpoint(self, sacType, flags, saccmd):

    # print('image data type:  ', dtype(self.imagedata))
    if flags[1]:  # change the image if its other than the saccade time...
        picidx = (self.interCounter % self.num_imagedata) + \
            1  # get the picture index
        self.interCounter += 1  # add one to the counter

    if flags[0]:  # if its other than the training time...
        # Random saccade selection    #-math.ceil(self.halfSizeCoarseSc[1])
        if sacType == 0:
            x = math.ceil(self.halfSizeCoarseSc[1])

            # getting random tx and ty
            # saccmd_x = random.randint(0,self.coarseScSize[1]-1)-math.ceil(self.halfSizeCoarseSc[1])   #coarsescsize is the size of the image 220*220. why ceil?
            # saccmd_y = random.randint(0,self.coarseScSize[0]-1)-math.ceil(self.halfSizeCoarseSc[0])

        if sacType == 3:  # For testing
            self.pos_x = np.asscalar(
                self.sequenceR[self.interC2+1]) % 336+1+152
            self.pos_Y = np.asscalar(
                self.sequenceR[self.interC2+2]) % 256+1+112
            self.interC2 += 1

        # self.pos_Limit()  # ensures that the fixation point is within the range/inside the range

    # of the saccade type is not 3 (which is correct/ type==0, random saccade)
    # if sacType != 3:
        # self.addDrift()

    return picidx


def take_file(self, filepath, imagedata):

    imagedata = scio.loadmat(filepath+"/LRimage1.mat")
    self.num_imagedata = len(imagedata)

    # print('image', type(self.image))
# Load whitening filter weights
    filterweights = scio.loadmat(filepath+"/wfilter55.mat")
    initialbases = scio.loadmat(
        filepath+"/initialbases.mat")  # Load initial bases
# Load transition probablity matrix
    transProb = scio.loadmat(filepath+"/transProb.mat")
# Load transition probablity matrix
    initialvergRLweights = scio.loadmat(filepath+"/initialweight.mat")
    self.train(0, imagedata)

    # model = self.config_model(imagedata['LRimage'], filterweights['wfilter'], initialbases['bases'],
    #                          transProb['transProb'], initialvergRLweights['weights'])  # Model configuration
# Environment.getLRwindows().self.imL
    # Training the model
    # self.train(0, model)
    # model.imagedata = []    # Empty the training data


# Save the trained model
    # timenow = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    # filename = filepath + "/data/VergenceLearning-" + timenow + ".pkl"
    # pickle.dump(model, open(filename, "wb"))
    # return imagedata

def train(self, imagedata, patch_size):

    for t in range(self.timeTrain):  # during the training time
        print('t:  ', t)

        # Calulate the coordinate of the next point and generate the input image batch
        # flag=array that includes other than training interval amd saccade interval
        flags = [not t % self.interval, not t %
                 (self.interval*self.saccadenum)]
        # the environmental model gets the fixed point....
        # picidx = self.getfixpoint(
        #    sacType, flags, self.saccmd)

        # Extract the sub-windows around the fixation point
        self.generate_randomTarCoord(imagedata, patch_size)

        # compute the correlation coefficient
        # corr = self.compute_corr_coef(
        #     t_x, t_y, w, eye_gaze_x, eye_gaze_y, im_BS, im_AS)

        # self.plot_img(t_x, t_y, eye_gaze_x,
        #               eye_gaze_y, imagedata, im_BS, im_AS, picidx)

        if not t % (self.timeTrain/100):
            print(str(t/(self.timeTrain/100))+'%\\ finished..')
