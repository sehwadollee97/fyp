from os import replace
import numpy as np
import pandas as pd
from sklearn import linear_model
import glob
import regex as re
import matplotlib.pyplot as plt
import csv

#linear factor scaling
#load the csv file 
class Camcalib_lp:
    def __init__(self, motor_init_x, motor_final_x, motor_init_y, motor_final_y):
        columnName=['x', 'y']
        self.columnName=columnName
        self.motor_init_x=motor_init_x
        self.motor_final_x=motor_final_x
        self.motor_delta_x=self.motor_final_x-motor_init_x

        self.motor_init_y=motor_init_y
        self.motor_final_y=motor_final_y
        self.motor_delta_y=self.motor_final_y-motor_init_y

        
        self.delta_motor_x=(pd.DataFrame({'x':np.repeat((self.motor_delta_x), (54))})).to_numpy()
        print('delta_motor_x', self.delta_motor_x)
        self.delta_motor_y=(pd.DataFrame({'y':np.repeat((self.motor_delta_y), (54))})).to_numpy()
        print('self.motor_delta_y', self.motor_delta_y)


         
    def import_file(self, filename):
        dataset=pd.read_csv(filename, names=self.columnName)
        dataset=pd.DataFrame(dataset)
        print('type print({}): '.format(dataset), type(dataset))
        return dataset
    
    def Extract_col(self, dataset):
        x=dataset["x"]
        y=dataset["y"]
        print('Extract column:  x column:  ', x)
        print('Extract column:  y column:  ', y)
        return x, y
        

    def AnotherFile(self, filename_2):
        dataset2=self.import_file(filename_2)
        x1, y1=self.Extract_col(dataset2)

        return x1, y1
        
        
    

    def computeDeltaPixel(self, x, x1, y, y1):
        self.delta_pixel_x=x1-x
        self.delta_pixel_y=y1-y
        print('self.delta_pixel_x:  ', self.delta_pixel_x)
        print('self.delta_pixel_y:  ', self.delta_pixel_y)

    def computeLinearFactors(self):

        self.a=np.divide(np.array(self.motor_delta_x), self.delta_pixel_x )
        self.b=np.divide(np.array(self.motor_delta_y), self.delta_pixel_y)
        print('a', self.a)
        print('b', self.b)
        print('type a_lp', type(self.a))
        print('type b_lp', type(self.b))
        print('shape a_lp', np.shape(self.a))
        print('shape b_lp', np.shape(self.b))

        #calculate the average of a and b
        a_mean=np.mean(self.a)
        b_mean=np.mean(self.b)

        print('a_lp_mean', a_mean)
        print('b_lp_mean', b_mean)

        return a_mean, b_mean



if __name__=='__main__':
    camcal=Camcalib_lp(425, 436, 717, 718 )
    dataset1=camcal.import_file('corners_updown717_left_pan425.csv')
    x, y=camcal.Extract_col(dataset1)
    x1, y1=camcal.AnotherFile('corners_updown717_left_pan436.csv')
    camcal.computeDeltaPixel(x, x1, y, y1)
    a_mean, b_mean=camcal.computeLinearFactors()


# right tilt a=-1.0942264472850682, b=1.7816286633368563
# left tilt a=-1.4229887456986126, b=1.8853966426007918

# left pan a=0.3693410005694254, b=-0.8143205148520528
#right pan a=-0.669696667088757, b=-0.33711135107519763

