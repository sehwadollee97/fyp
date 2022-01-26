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
    from Agent import Agent
    #from PygameRobot import PygameRobot
    import pygame
    from CamControl import CamControl
    from EyeMotor import EyeMotor
    from AECControl import AECControl
    import matplotlib.patches as patches
    #filepath = os.path.dirname(os.path.realpath(__file__))
    #print('filepath', filepath)

    filepath='/home'

    from AEC_vision.aec_vision.utils.simulation import Simulation
    from sklearn.metrics import mean_squared_error
    from AEC_vision.aec_vision.utils.environment import Environment
    from AEC_vision.aec_vision.utils.utils import get_cropped_image
    from AEC_vision.aec_vision.utils.plots import plot_observations
    from AEC_vision.aec_vision.utils.graphics_math.constant_tools import deg_to_rad, rad_to_deg
    
    #print(os.path.join(filepath,'/fyp/Downloads/SCServo_Python_200831/SCServo_Python',"AEC-vision", "/aec-vision/", "utils"))

except Exception as e:
    print('Some modules are missing {}'.format(e))

filepath = os.path.dirname(os.path.realpath(__file__))
print('filepath', filepath)

# decicde robot
robotype = 3
# rotation matrix- in degree
# rx = math.radians(0)
# ry = math.radians(0)
# rz = math.radians(0)
# rx_deg = round(math.degrees(rx))
# ry_deg = round(math.degrees(ry))
# rz_deg = round(math.degrees(rz))
# # translation matrix
# tx = 0
# ty = 0
# tz = 0

patch_size = 80

# instantiate agent
agent = Agent(10)

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

# CamMat_left=np.array([[429.81312592,   0.,         355.63374988], 
#  [  0.,         407.84669903, 199.28138355], 
#  [  0.,           0.,           1.        ]])


# CamMat_right=np.array([
# [315.49620658,   0.,         310.038859  ],
#  [  0.   ,      316.30375674, 239.97495082],
#  [  0.  ,         0. ,          1.        ]
# ])



left = Sophia(14)
right = Sophia(15)
updown = Sophia(16)

# from camcalib class
pan_left_scale_a=0.3693410005694254
pan_left_scale_b=-0.8143205148520528
pan_right_scale_a=0.669696667088757
pan_right_scale_b=-0.33711135107519763

tilt_left_scale_a=-1.4229887456986126
tilt_left_scale_b=1.8853966426007918
tilt_right_scale_a=-1.0942264472850682
tilt_right_scale_b=1.7816286633368563

####################################################################################
#calibration

if robotype==2:

    

    

    # _, img_left_coarse, _, img_right_coarse = simulation.get_observations()
    # disparity = mean_squared_error(img_left_coarse, img_right_coarse)
    # plot_observations(img_left_coarse, img_right_coarse, texture_dist=texture_dist,
    #             camera_angle=simulation.environment.camera_angle, disparity=disparity)

    

    while True:

        #image generation from the robot

        texture_dist = 500
        simulation = Simulation(action_to_angle=[-1,0,1])
        simulation.environment.add_texture(dist=texture_dist)
        img, _ = simulation.environment.render_scene()
        # plt.imshow(img)
        # plt.show()

        simulation.new_episode(texture_dist=texture_dist, texture_file='data/texture1.jpg')
        img1, _ = simulation.environment.render_scene()
        # plt.imshow(img1)
        # plt.show()

        img_dim_x, img_dim_y=agent.ImageProcessing(img1, patch_size, 2)

        
        print('image dimension x_right:  ', img_dim_x)
        print('image dimension y_right:  ', img_dim_y)

        # generate random coordinates on the image  & ensures
        agent.generate_randomTargetCoord()

        pan, tilt= agent.generate_eyeCmd(pan_left_scale_a, pan_left_scale_b, pan_right_scale_a,pan_right_scale_b,
        tilt_left_scale_a,tilt_left_scale_b,  tilt_right_scale_a, tilt_right_scale_b)

        # for now, 
        # keep the tilt=0
    
        pan_min=-2500
        pan_max=2500
        tilt_min=-2500
        tilt_max=2500

        camera_angle = 0.5
        pan_valid, tilt_valid=simulation.check_the_limits(camera_angle, tilt,pan_min, pan_max, tilt_min, tilt_max)
        agent.compute_saccade_positions(pan_valid, tilt_valid)
        # show image center
        #show the tx_ty position 
        #caqpture the image 
        agent.plot_images(img1)
        img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = simulation.get_observations()

        simulation.environment.move_camera(camera_angle)
        print('after moving camera')
        #show image center
        # show how much camera angle rotation leads to how much shift in the image pixel
        # 
        #agent.plot_images(img1)


        # have to capture the image around the 
        img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = simulation.get_observations()


        disparity = mean_squared_error(img_left_fine, img_right_fine)
        plot_observations(img_left_fine, img_right_fine, texture_dist=texture_dist,
                          camera_angle=simulation.environment.camera_angle, disparity=disparity)
        #agent.plot_images(img_left_coarse)
        #setting the value of the pan_max,min and tilt max and min


       

        # originalimg, im_BS, im_AS, corr, counter=agent.plot_images(right_img_as)
        

        # print('correlation coefficient:  ', corr)



if robotype==3:

    
    #min_left = left.calibration(0)
    #max_left = left.calibration(1)
    #left.move((min_left+max_left)/2)
    # _, _, mid_l=left.get_limit()
    # left.move(mid_l)



    # #min_right = right.calibration(0)
    # #max_right = right.calibration(1)
    # #right.move((min_right+max_right)/2)
    # _, _, mid_r=right.get_limit()
    # right.move(mid_r)


    # # min_updown = updown.calibration(0)
    # # max_updown = updown.calibration(1)
    # # updown.move((min_updown+max_updown)/2)
    # _, _, mid_u=updown.get_limit()
    # updown.move(mid_u)


    # print('min left: {0} & max left:{1}  '.format(min_left,max_left) )
    # # print('min right: {0} & max right: {1}'.format( min_right,max_right))
    # print('min updown: {0}, max_updown: {1}'.format(min_updown,max_updown))

    #obain the current position of the left and right
    left_currentpos=left.get_currentPos()
    right_currentpos=right.get_currentPos()
    updown_currentpos=updown.get_currentPos()

    eye_type=['L', 'R']
    while True:
        aec=AECControl()

        # image capture
        left_img, right_img=aec.capture()
        # aec.right.release()
        # aec.left.release()
     

        #current position of left and right eye
        currPos_left=left.get_currentPos()
        currPos_right=right.get_currentPos()
        currPos_updown=updown.get_currentPos()

        #getting min, max and mid values for left and right eyes-pan & tilt

        min_left, max_left, mid_left=left.get_limit()
        min_right, max_right, mid_right=right.get_limit()
        min_updown, max_updown, mid_updown=updown.get_limit()

        #!!!!!!!!!!!!!!!!! have to decide left or right
        img_dim_x_right, img_dim_y_right=agent.ImageProcessing(left_img, patch_size, robotype)
        print('image dimension x_right:  ', img_dim_x_right)
        print('image dimension y_right:  ', img_dim_y_right)

        # img_dim_x_left, img_dim_y_left=agent.ImageProcessing(left_img, patch_size, robotype)
        # print('image dimension x_left:  ', img_dim_x_left)
        # print('image dimension y_left:  ', img_dim_y_left)


        # right tilt a=-1.0942264472850682, b=1.7816286633368563
        # left tilt a=-1.4229887456986126, b=1.8853966426007918

        # left pan a=0.3693410005694254, b=-0.8143205148520528
        #right pan a=-0.669696667088757, b=-0.33711135107519763

        # from camcalib class

        #left
#  average ab 
#  [[-0.13071265]
#  [ 0.31192909]]
# average cd 
#  [[-0.8942854 ]
#  [-0.27277954]]

#right 

# average ab 
#  [[-0.14171019]
#  [ 0.06591771]]
# average cd 
#  [[-0.50439132]
#  [-0.09609053]]
        # pan_left_scale_a=0.3693410005694254
        # pan_left_scale_b=-0.8143205148520528
        # pan_right_scale_a=0.669696667088757
        # pan_right_scale_b=-0.33711135107519763

        # tilt_left_scale_a=-1.4229887456986126
        # tilt_left_scale_b=1.8853966426007918
        # tilt_right_scale_a=-1.0942264472850682
        # tilt_right_scale_b=1.7816286633368563

                #left
        #average ab 
        left_ab=np.array([[-0.13071265],[ 0.31192909]])
        print('left_ab', left_ab[0])
        print('shape of left_ab', np.shape(left_ab))
    #average cd 
        left_cd=np.array([[-0.8942854 ],
        [0.27277954]])


           # average ab 
        right_ab=np.array([[-0.14171019],
        [ 0.06591771]])
        #average cd 
        right_cd=np.array([[-0.50439132],
        [-0.09609053]] )



        # generate random coordinates on the image  & ensures
        agent.generate_randomTargetCoord()
        pan, tilt= agent.generate_eyeCmd(left_ab, left_cd, right_ab, right_cd)

        print('pan:   ', pan)
        print('tilt:   ', tilt)

        #setting the value of the pan_max,min and tilt max and min
        pan_min = -100
        pan_max = 150
        tilt_min = -100
        tilt_max = 150

        # veryfying the limits of he left eye in sophia
        # pan_valid, tilt_valid = Left_eye.check_the_limits(
        #     pan, tilt,cur_pos_left, cur_pos_updown, min_left, max_left, min_updown, max_updown)

        pan_valid, tilt_valid = right.check_the_limits(
            pan, tilt,pan_min, pan_max, tilt_min, tilt_max)
        
        # pan_valid, tilt_valid = left.check_the_limits(
        #     pan, tilt,pan_min, pan_max, tilt_min, tilt_max)


        pan_valid, tilt_valid=agent.compute_saccade_positions(pan_valid, tilt_valid)
        #displacement_x=int(round(math.tan(pan_valid)*fx))
        #displacement_y=int(round(math.tan(tilt_valid)*fy))

        print('directional pan_valid:  ', pan_valid)
        print('directional tilt_valid:  ', tilt_valid)

        #BS_position=left.get_currentPos()
        #print('left_BS_position', BS_position)

        print('moving the pan')
        cur_pos_left=left.moveby(pan_valid) # move by 

        #AS_position=left.get_currentPos()
        #print('left_AS_position', AS_position)

        #updown_BS_position=updown.get_currentPos()
        #print('updown_BS_position', updown_BS_position)

        print('moving the tilt')
        cur_pos_updown=updown.moveby(tilt_valid)

        #updown_AS_position=updown.get_currentPos()
        #print('updown_AS_position', updown_AS_position)
        print('capture!')
        


        left_img_as, right_img_as=aec.capture()
        aec.right.release()
        aec.left.release()



        agent.plot_images(left_img_as)
        

        #print('correlation coefficient:  ', corr)




#print('current left: {0}, current right: {1}, current updown: {2}'.format( left_currentpos, right_currentpos, updown_currentpos))

####################################################################################



# print('image type', type(right_img))
# print('image type', type(left_img))
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

# motor_unit_pan_left_init=419
# motor_unit_pan_left_final=429

# pixel_pan_left_init_x=155
# pixel_pan_left_init_y=67.9

# pixel_pan_left_final_x=192.6
# pixel_pan_left_final_y=68.6

# #pan left scale
# pan_left_scale=abs(pixel_pan_left_final_x-pixel_pan_left_init_x)/abs(motor_unit_pan_left_final-motor_unit_pan_left_init)
# print('pan left scale:  ', pan_left_scale)


# #pan-right
# motor_unit_pan_right_init=500
# motor_unit_pan_right_final=510

# pixel_pan_right_init_x=173.8
# pixel_pan_right_init_y=22.5

# pixel_pan_right_final_x=208.9
# pixel_pan_right_final_y=23.8


# pan_right_scale=abs(pixel_pan_right_final_x-pixel_pan_right_init_x)/abs(motor_unit_pan_right_final-motor_unit_pan_right_init)
# print('pan right scale:  ', pan_right_scale)



#tilt-left



# motor_unit_tilt_left_init=600
# motor_unit_tilt_left_final=610

# pixel_tilt_left_init_x=202
# pixel_tilt_left_init_y=67.9

# pixel_tilt_left_final_x=200
# pixel_tilt_left_final_y=74.4

# #tilt-right
# motor_unit_tilt_right_init=600
# motor_unit_tilt_right_final=610

# pixel_tilt_right_init_x=178.3
# pixel_tilt_right_init_y=39.4

# pixel_tilt_right_final_x=179.6
# pixel_tilt_right_final_y=21.2



# tilt_left_scale=abs(pixel_tilt_right_final_y-pixel_tilt_right_init_y)/abs(motor_unit_tilt_right_final-motor_unit_tilt_right_init)
# print('tilt left scale:  ', tilt_left_scale)





# tilt_right_scale=abs(pixel_tilt_right_final_y-pixel_tilt_right_init_y)/abs(motor_unit_tilt_right_final-motor_unit_tilt_right_init)
# print('tilt right scale:  ', tilt_right_scale)




#########################################
# fx=315.49620658
# fy=316.30375674

