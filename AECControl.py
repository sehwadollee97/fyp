import cv2
import numpy as np
import matplotlib.pyplot as plt
# cap = cv2.VideoCapture(2) #right looking from the robot 
# cap2 = cv2.VideoCapture(0) #left looking from the robot
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# '''
#  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#  cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#  cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# '''
# cv2.namedWindow('test1',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
# cv2.namedWindow('test2',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# while(True):
#     ret1, frame1 = cap.read()
#     ret2, frame2 = cap2.read()

#     # faces1 = face_cascade.detectMultiScale(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
#     # faces2 = face_cascade.detectMultiScale(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY))
    
#     # for (x, y, w, h) in faces1:
#     #     cv2.rectangle(frame1, (x,y), (x+w, y+h), (255,0,0), 2)
#     #     subframe1 = cv2.cvtColor(frame1[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
#     #     eyes = eyes_cascade.detectMultiScale(subframe1)
#     #     for (ex, ey, ew, eh) in eyes:
#     #         cv2.rectangle(frame1, (x + ex, y + ey),(x + ex + ew, y + ey + eh), (0,255,0),2)
            
#     # for (x, y, w, h) in faces2:
#     #     cv2.rectangle(frame2, (x,y), (x+w, y+h), (255,0,0), 2)
#     #     subframe2 = cv2.cvtColor(frame2[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
#     #     eyes = eyes_cascade.detectMultiScale(subframe1)
#     #     for (ex, ey, ew, eh) in eyes:
#     #         cv2.rectangle(frame2, (x + ex, y + ey),(x + ex + ew, y + ey + eh), (0,255,0),2)
    
#     cv2.imshow('test1',frame1)
#     cv2.imshow('test2',frame2)
#     key = cv2.waitKey(20) # mili second
    
#     if key == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


# The class for AEC Code

class AECControl:

    def __init__(self, width = 320, height = 240):

        # video stream capture for each eye
        self.right = cv2.VideoCapture(0) #right eye
        self.left = cv2.VideoCapture(2) #left eye

        # set the image width and height
        self.right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.left.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    
    # return image captured by the two eyes
    def capture(self):
        #right_result = left_result = False
        #while right_result == False or left_result == False:
        right_result , frame_right = self.right.read()
        left_result , frame_left = self.left.read()
        #print("running",right_result,left_result)
        return frame_right, frame_left

