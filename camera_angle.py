import numpy as np
import math


class camera_angle:

    def __init__(self):
        pass

    def compute_R(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

        Rx = np.array(
            [[1, 0, 0],
             [0, math.cos(self.a), (math.sin(self.a))],
                [0, -1*math.sin(self.a), math.cos(self.a)]])

        Rx = np.float32(Rx)
        #print('Rx:   ', Rx)

        Ry = np.array(
            [[math.cos(self.b), 0, -1*math.sin(self.b)],
             [0, 1, 0],
                [math.sin(self.b), 0, math.cos(self.b)]])
        Ry = np.float32(Ry)
        #print('Ry:   ', Ry)

        Rz = np.array(
            [[math.cos(self.c), math.sin(self.c), 0],
             [-1*math.sin(self.c), math.cos(self.c), 0],
                [0, 0, 1]])
        Rz = np.float32(Rz)
        #print('Rz:   ', Rz)

        interme = np.matmul(Rx, Ry)
        Rm = np.matmul(interme, Rz)
        return Rm


'''

class compute_R:
    def __init__(self, R):




    
#R=Rx(a)*Ry(b)*Rz(c)

'''
