import pygame
from pygame.locals import *
#from pyrender import camera
from pyrr import Vector3, Matrix44, vector, vector3
from math import sin, cos, radians
import cv2
import pickle
import time
import os
import math
import numpy as np
import random

import scipy.io as scio
import matplotlib.pyplot as plt

# environment
from numpy.lib.shape_base import expand_dims
#from skimage.util import view_as_windows

from camera_angle import camera_angle


class CamControl:

    def __init__(self):
        self.camera_pos = Vector3([0.0, 0.0, 0.0])
        self.camera_front = Vector3([0.0, 0.0, -1.0])
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        self.camera_right = Vector3([1.0, 0.0, 0.0])

        self.mouse_sensitivity = 0.5
        self.velocity = 0.05
        self.yaw = 0.0
        self.pitch = 0.0

        self.lastX = 64
        self.lastY = 64

        self.pan_valid = 0.0
        self.tilt_valid = 0.0

    # Use keys to control the camera translation,
    # Use mouse to control the camera rotation.

    def compute_cameramat(self,  rx, ry, rz, tx, ty, tz):
        # this computes the rotation matrix
        self.r_matrix = camera_angle()
        self.Rm = self.r_matrix.compute_R(rx, ry, rz)
    # translation matrix
        self.t = np.array([[tx, ty, tz, 1]]).T   # x y z
        self.zero_matrix = np.array([[0, 0, 0]])

        # concatenate in the x axis
        self.R_t = np.concatenate((self.Rm, self.zero_matrix), axis=0)
        # concatenate in the y axis
        self.cam_pose = np.concatenate((self.R_t, self.t), axis=1)
        print('cam_pose:  ', self.cam_pose)
        return self.cam_pose

    def capture_image(self):

        pass

    def getviewmat_keymouseCtrl(self, pan_valid, tilt_valid):
        self.pan_valid = pan_valid
        self.tilt_valid = tilt_valid
        #pressedkey = pygame.key.get_pressed()
        #mousepos = pygame.mouse.get_pos()
        # use of keyboard input to control the camera translation
        # self.process_key_movement()
        self.process_mouse_movement()
        return self.look_at()

    # Use keys to control the camera translation,
    # The input commands control the camera rotation.

    ######## this function is not used####################################

    def getviewmat_keyTranslationCtrl(self, yawcmd, pitchcmd):
        pressedkey = pygame.key.get_pressed()
        # use of keyboard input to control the camera translation
        self.process_key_movement(pressedkey)

        self.yaw = yawcmd
        self.pitch = pitchcmd
        self.update_camera_vectors()
        return self.look_at()
    ######## ##################################################################

    # The camera position is fixed at the origin,
    # The input commands control the camera rotation.
    def getviewmat_CmdCtrl(self, yawcmd, pitchcmd):
        self.yaw = yawcmd
        self.pitch = pitchcmd
        self.update_camera_vectors()
        return self.look_at()

    def process_key_movement(self):
        pass

        # # use of keyboard input to control the camera translation
        # if pressedkey[K_w]:
        #     self.camera_pos += self.camera_front * self.velocity
        # if pressedkey[K_s]:
        #     self.camera_pos -= self.camera_front * self.velocity
        # if pressedkey[K_d]:
        #     self.camera_pos += self.camera_right * self.velocity
        # if pressedkey[K_a]:
        #     self.camera_pos -= self.camera_right * self.velocity
        # if pressedkey[K_q]:
        #     self.camera_pos += self.camera_up * self.velocity
        # if pressedkey[K_e]:
        #     self.camera_pos -= self.camera_up * self.velocity

    def process_mouse_movement(self):
        # set the mouse position x and y

        xpos = self.pan_valid
        ypos = self.tilt_valid
        # print(xpos, ypos)

        # assign the yaw an pitch
        self.yaw = (xpos - self.lastX)*self.mouse_sensitivity
        self.pitch = (self.lastY - ypos)*self.mouse_sensitivity

        self.update_camera_vectors()

    def update_camera_vectors(self):
        # update the camera vectors by calculating the yaw and pitch
        front = Vector3([0.0, 0.0, 0.0])
        front.x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        front.y = sin(radians(self.pitch))
        front.z = sin(radians(self.yaw)) * cos(radians(self.pitch))

        # calculate the camera_fron, right and up by normalization
        self.camera_front = vector.normalise(front)
        self.camera_right = vector.normalise(vector3.cross(
            self.camera_front, Vector3([0.0, 1.0, 0.0])))
        self.camera_up = vector.normalise(
            vector3.cross(self.camera_right, self.camera_front))

    def look_at(self):

        # final computation of the all the intermediates
        position = self.camera_pos
        target = self.camera_pos + self.camera_front
        world_up = self.camera_up

        # 1.Position = known
        # 2.Calculate cameraDirection
        zaxis = vector.normalise(position - target)
        # 3.Get positive right axis vector
        xaxis = vector.normalise(vector3.cross(
            vector.normalise(world_up), zaxis))
        # 4.Calculate the camera up vector
        yaxis = vector3.cross(zaxis, xaxis)

        # create translation and rotation matrix
        translation = Matrix44.identity()
        translation[3][0] = -position.x
        translation[3][1] = -position.y
        translation[3][2] = -position.z

        rotation = Matrix44.identity()
        rotation[0][0] = xaxis[0]
        rotation[1][0] = xaxis[1]
        rotation[2][0] = xaxis[2]
        rotation[0][1] = yaxis[0]
        rotation[1][1] = yaxis[1]
        rotation[2][1] = yaxis[2]
        rotation[0][2] = zaxis[0]
        rotation[1][2] = zaxis[1]
        rotation[2][2] = zaxis[2]

        return rotation * translation

    def check_the_limits(self, pan, tilt, pan_min, pan_max, tilt_min, tilt_max):
        print('type of pan_min before floating:  ', type(pan_min))
        print('type of pan_max before floating:  ', type(pan_max))
        pan_min = float(pan_min)
        pan_max = float(pan_max)
        print('type of pan_min after floating:  ', type(pan_min))
        print('type of pan_max after floating:  ', type(pan_max))

        if pan < pan_min or pan > pan_max:
            if pan < pan_min:
                pan_valid = pan_min
            elif pan > pan_max:
                pan_valid = pan_max

        else:
            pan_valid = pan
            print('pan is within the range')

        if tilt < tilt_min or tilt > tilt_max:
            if tilt < tilt_min:
                tilt_valid = tilt_min
            elif tilt > tilt_max:
                tilt_valid = tilt_max
        else:
            tilt_valid = tilt

        return pan_valid, tilt_valid
