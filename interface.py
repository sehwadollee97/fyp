# interface-imports classes of robots
# instantiates the robots
# instantiates the agent
# write the base class




try:
    import pickle
    import time
    import os
    from os import stat
    import scipy.io as scio
    import matplotlib.pyplot as plt
    #import matplotlib.pyplot as plt
    import sys
    import numpy as np
    import math
    import cv2
    from PIL import Image
    from sophia import Sophia

    #from camera_angle import camera_angle

    # from Simulated_robot import Simulated_robot  # import robot
    #from PyrenderRobot import PyrenderRobot
    # from Demo import Demo  # import saccade
    #from Staticimages import Staticimages
    from Agent import Agent
    #from PygameRobot import PygameRobot
    import pygame
    from CamControl import CamControl
    from EyeMotor import EyeMotor
    from AECControl import AECControl


except Exception as e:
    print('Some modules are missing {}'.format(e))

filepath = os.path.dirname(os.path.realpath(__file__))
print('filepath', filepath)

# decicde robot
robotype = 3
# rotation matrix- in degree
rx = math.radians(0)
ry = math.radians(0)
rz = math.radians(0)


rx_deg = round(math.degrees(rx))
ry_deg = round(math.degrees(ry))
rz_deg = round(math.degrees(rz))

# translation matrix
tx = 0
ty = 0
tz = 0

patch_size = 80





# instantiate agent
agent = Agent()


# aec = AECControl(1920,1080)
# right_img , left_img = aec.capture()
# print('image type', type(right_img))
# print('image type', type(left_img))
# aec.right.release()
# aec.left.release()
# print('image type', type(right_img))
# print('image type', type(left_img))
# print(np.shape(right_img))

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(right_img)

# plt.subplot(1, 2, 2)
# plt.imshow(left_img)
# plt.show()

camctrl = CamControl()  # instantiate the cameral control

#camera matraix for left eye

CamMat_left=np.array([[429.81312592,   0.,         355.63374988], 
 [  0.,         407.84669903, 199.28138355], 
 [  0.,           0.,           1.        ]])


CamMat_right=np.array([
[315.49620658,   0.,         310.038859  ],
 [  0.   ,      316.30375674, 239.97495082],
 [  0.  ,         0. ,          1.        ]
])

Left_eye = Sophia(14)
Right_eye = Sophia(15)
Updown_eye = Sophia(16)
original_left=419
original_updown=419

####################################################
# calibration

# min_right = Right_eye.calibration(0)
# max_right = Right_eye.calibration(1)


# cur_pos_right=Right_eye.move((min_right+max_right)/2)
# print(min_right,max_right, cur_pos_right)


#left
# min_left = Left_eye.calibration(0)
# max_left = Left_eye.calibration(1)

cur_pos_left=Left_eye.move(419)
# print(min_left,max_left, cur_pos_left)

# #updown
# min_updown = Updown_eye.calibration(0)
# max_updown = Updown_eye.calibration(1)


# cur_pos_updown=Updown_eye.move((min_updown+max_updown)/2)
# print(min_updown,max_updown, cur_pos_updown)


# Updown_eye.move(original_updown, False)
aec=AECControl()
left_img, right_img=aec.capture()


print('image type', type(right_img))
print('image type', type(left_img))
#Left_eye.right.release()
#Right_eye.left.release()

# print(np.shape(right_img))

# plt.figure()
# #plt.subplot(1, 2, 1)
# plt.imshow(left_img)

# #plt.subplot(1, 2, 2)
# #plt.imshow(right_img)
# plt.show()

# motor-pixel calibration
#pan-left

motor_unit_pan_left_init=419
motor_unit_pan_left_final=429

pixel_pan_left_init_x=155
pixel_pan_left_init_y=67.9

pixel_pan_left_final_x=192.6
pixel_pan_left_final_y=68.6


pan_left_scale=abs(pixel_pan_left_final_x-pixel_pan_left_init_x)/abs(motor_unit_pan_left_final-motor_unit_pan_left_init)
print('pan left scale:  ', pan_left_scale)


#pan-right
motor_unit_pan_right_init=500
motor_unit_pan_right_final=510

pixel_pan_right_init_x=173.8
pixel_pan_right_init_y=22.5

pixel_pan_right_final_x=208.9
pixel_pan_right_final_y=23.8


pan_right_scale=abs(pixel_pan_right_final_x-pixel_pan_right_init_x)/abs(motor_unit_pan_right_final-motor_unit_pan_right_init)
print('pan right scale:  ', pan_right_scale)



#tilt-left

motor_unit_tilt_left_init=600
motor_unit_tilt_left_final=610

pixel_tilt_left_init_x=202
pixel_tilt_left_init_y=67.9

pixel_tilt_left_final_x=200
pixel_tilt_left_final_y=74.4


tilt_left_scale=abs(pixel_tilt_left_final_y-pixel_tilt_left_init_y)/abs(motor_unit_tilt_left_final-motor_unit_tilt_left_init)
print('tilt left scale:  ', tilt_left_scale)


#tilt-right
motor_unit_tilt_right_init=600
motor_unit_tilt_right_final=610

pixel_tilt_right_init_x=178.3
pixel_tilt_right_init_y=39.4

pixel_tilt_right_final_x=179.6
pixel_tilt_right_final_y=21.2


tilt_right_scale=abs(pixel_tilt_right_final_y-pixel_tilt_right_init_y)/abs(motor_unit_tilt_right_final-motor_unit_tilt_right_init)
print('tilt right scale:  ', tilt_right_scale)





# loading the image


# simulated robot gets the image
#image = simRobot.main(rx, ry, rz, tx, ty, tz)
#print('image', image)

#imageName = 'Screenshot {counter} with rx={rx}, ry={ry}, rz={rz}, tx={tx}, ty={ty}, tz={tz}.png'.format(
#    counter=1, rx=rx_deg, ry=ry_deg, rz=rz_deg, tx=tx, ty=ty, tz=tz)
#print('filepath', filepath)


#image = Image.open(filepath+'/imagebrick.png')
# image.show()



#########################################
fx=315.49620658
fy=316.30375674


img_dim_x, img_dim_y=agent.ImageProcessing(left_img, patch_size, fx, fy, robotype)
agent.generate_randomTargetCoord()
pan, tilt, tx, ty = agent.generate_eyeCmd(pan_left_scale, pan_right_scale, tilt_left_scale, tilt_right_scale)

print('pan:   ', pan)
print('tilt:   ', tilt)


#setting the value of the pan_max,min and tilt max and min
pan_min = -150
pan_max = 150
tilt_min = -100
tilt_max = 100

# veryfying the limits of he left eye in sophia
# pan_valid, tilt_valid = Left_eye.check_the_limits(
#     pan, tilt,cur_pos_left, cur_pos_updown, min_left, max_left, min_updown, max_updown)

pan_valid, tilt_valid = Left_eye.check_the_limits(
    pan, tilt, pan_min, pan_max, tilt_min, tilt_max)




pan_valid, tilt_valid=agent.compute_saccade_positions(pan_valid, tilt_valid)
#displacement_x=int(round(math.tan(pan_valid)*fx))
#displacement_y=int(round(math.tan(tilt_valid)*fy))

print('directional pan_valid:  ', pan_valid)
print('directional tilt_valid:  ', tilt_valid)


print('moving the pan')
cur_pos_left=Left_eye.moveby(pan_valid) # move by 

print('moving the tilt')
cur_pos_updown=Updown_eye.moveby(tilt_valid)


left_img_as, right_img_as=aec.capture()
aec.right.release()
aec.left.release()
# plt.figure()
# plt.imshow(left_img_as)
# plt.show()

agent.plot_images(left_img_as)



# to be arranged by the fyp group
#imagedata = scio.loadmat(filepath+"/LRimage1.mat")

#while True:


'''

        #pass on the left image to the agent
    agent.ImageProcessing(left_img, patch_size, f, robotype)
    agent.generate_randomTargetCoord()
    pan, tilt = agent.generate_eyeCmd()  # telling sophia how much its eyes have to move

    pan_valid, tilt_valid = sophia.check_the_limits(
        pan, tilt, pan_min, pan_max, tilt_min, tilt_max)
# sophia move the camera function
    sophia.Move_the_camera(pan_valid, tilt_valid, rz, tx, ty, tz)
# capture the image

    # sophia.Setup_parameters(16, 534, 534)
    # sophia.Execute_motorcmd()

    # #captures the image accordingly
    # right_img , left_img = sophia.captureImg()

    agent.plot_img(pan_valid, tilt_valid)
    # agent.compute_corr_coef()

'''
