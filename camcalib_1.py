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

columnName=['x', 'y']
dataset_leftpan_425=pd.read_csv('corners_updown717_left_pan425.csv', names=columnName)


dataset_leftpan_425=pd.DataFrame(dataset_leftpan_425)
print('type print(dataset_leftpan_425): ', type(dataset_leftpan_425))
#print(dataset_leftpan_425)

x_lp_425=dataset_leftpan_425["x"]
#print('type', type(dataset_leftpan_425["x"]))
#x_lp_425=x_lp_425.str.split(' ', expand = True)
#x_lp_425=x_lp_425.dropna()


#print('x_lp_425', x_lp_425)
y_lp_425=dataset_leftpan_425["y"]
#print('y_lp_425', y_lp_425)



dataset_leftpan_436=pd.read_csv('corners_updown717_left_pan436.csv', names=columnName)


dataset_leftpan_436=pd.DataFrame(dataset_leftpan_436)
print('type print(dataset_leftpan_425): ', type(dataset_leftpan_436))
#print(dataset_leftpan_436)

x_lp_436=dataset_leftpan_436["x"]
print('x_lp_425', x_lp_436)



y_lp_436=dataset_leftpan_436["y"]
print('y_lp_436', y_lp_436)



# compute change in x and change in y 

delta_pixel_x_lp=x_lp_425-x_lp_436
delta_pixel_y_lp=y_lp_425-y_lp_436


#print('delta_pixel_x_lp',delta_pixel_x_lp )
#print('delta_pixel_y_lp', delta_pixel_y_lp)
q=436-425
delta_motor_x_lp=pd.DataFrame({'x':np.repeat((436-425), (54))})
print('delta_motor_x_lp', delta_motor_x_lp)
delta_motor_y_lp=pd.DataFrame({'x':np.repeat((717-716), (54))})
#print('delta_motor_y_lp', delta_motor_y_lp)



delta_motor_x_lp=delta_motor_x_lp.to_numpy()
delta_motor_y_lp=delta_motor_y_lp.to_numpy()

delta_pixel_x_lp=delta_pixel_x_lp.to_numpy()
delta_pixel_y_lp=delta_pixel_y_lp.to_numpy()

a_lp=np.divide(np.array(436-425), delta_pixel_x_lp )
b_lp=np.divide(np.array(717-716), delta_pixel_y_lp)
print('a', a_lp)
print('b', b_lp)
print('type a_lp', type(a_lp))
print('type b_lp', type(b_lp))
print('shape a_lp', np.shape(a_lp))
print('shape b_lp', np.shape(b_lp))

#calculate the average of a and b
a_lp_mean=np.mean(a_lp)
b_lp_mean=np.mean(b_lp)

print('a_lp_mean', a_lp_mean)
print('b_lp_mean', b_lp_mean)

print('a_lp_mean shape', np.shape(a_lp_mean))
print('b_lp_mean shape', np.shape(b_lp_mean))




#compute linear regression of a




# use




#read corners txt file for left 
#save them as numpy




# with open('corners_left_pan448.txt') as file:
#     arr_left_pan437 = np.array(
#         [
#             np.array([str(num) for num in line.lstrip('[[ ').rstrip(' ]]').split(",")]) 
#             for line in file
            
            


#         ]
#     )


# list1=arr_left_pan437.tolist()
# for lines in list1:
#     lines.rstrip(']]\n', '')
#     print('lines', lines)

#     #print('list1',list1)


#     # for line in arr_left_pan437:
        
#     #     line=np.float(line)





# #data=np.random.randn(3000000,50)
# #print(type(data[1][1])) # numpy.float64
# print(arr_left_pan437)
# print('np.shape(arr_left_pan437',np.shape(arr_left_pan437))
# print('dype',type(arr_left_pan437))
# print('dtype of element', type(arr_left_pan437[53][0]))
# #arr_left_pan437=arr_left_pan437.astype(np.float)
# #print('dype',type(arr_left_pan437))




# #np.hsplit(arr_left_pan437, 1)
# print(arr_left_pan437)
# x=arr_left_pan437[:, 0]
# print('x', x)

# print('size of x', np.shape(x))

# y=arr_left_pan437[:, 1]
# print('y', y)
# print('size of y', np.shape(y))

# #converting the x and y into float 
# print(range(np.size(x)-1))
# for i in range(np.size(x)-1):
#     x[i]=float(x[i])

# for i in range(np.size(y)-1):
#     y[i]=float(y[i])
# print(type(y), type(x))
# print(np.shape(y), np.shape(x))

# data=np.stack((x, y))
# data=data.T
# print(len(data.T))





# regr=linear_model.LinearRegression()
# #regr.fit(x, y)

# plt.figure()
# plt.plot(x, y)
# plt.show()

# df=pd.DataFrame(data, columns=['x', 'y'])
# print(df)








# with open('corners_left_pan437.txt') as file:

#     arr_left_pan437 = np.array(
#         [
#             np.array([str(num) for num in line.lstrip('[[ ').rstrip(' ]]').split(",")]) 
#             for line in file
            
            


#         ]
#     )

# filenames_left_pan=glob.glob('./corners_left_pan???.txt')
# print(filenames_left_pan)
# filenames_right_pan=np.load('corners_right_pan425.txt')
# for left_pan in filenames_right_pan:
#     for lines in left_pan:
#         with open(left_pan, 'r') as file:
#                 lines.rstrip(']]').lstrip('[[')
#                 print(lines)
                #file.write(left_pan)

        # with open(left_pan) as file:
        #     lines=np.loadtxt(left_pan)
        #     print(lines)

        # arr=np.array(left_pan)
        # print('arr', arr)
        # print('type', type(arr))

        # print(   )
        #print('shape', len(arr))
        

        # if arr!=0:
        #     print(True)
        



# with open('corners_right_pan425.txt') as f:
#     lines=f.readlines()
    
# #     bj=re.search('(\d+)',lines)
# #     print(bj)
#     print('type', type(lines))

#     arr=np.array(
#             [
#                      np.array([str(num) for num in line.lstrip('[[ ').rstrip(' ]]').split(",")])
#                      for line in f
#             ]
#     )
#     print(arr)
    

# with open('corners_right_pan425.txt') as file:
#     arr_left_pan437 = np.array(
#         [
#             np.array([str(num) for num in line.lstrip('[[ ').rstrip(' ]]').split(",")]) 
#             for line in file
            
            


#         ]
#     )
#     print(arr_left_pan437)

#read_file=pd.read_csv(r'corners_right_pan527.txt')
#read_file.to_csv(r'corners_right_pan527.csv', index=None)

# filename=open('corners_right_pan425.csv')
# csvreader=csv.reader(filename, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
# header=[]
# header=next(csvreader)
# print(header)


# def loadCsv(filename):
#     with open(filename,'r') as f:
#         f.readline() # skip header

#         csvf = csv.reader(f)
#         dataset = [[float(x) for x in row] for row in csvf]
#         return dataset