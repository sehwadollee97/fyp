#!/usr/bin/env python
#
# *********     try get load     *********
#
# Trying to write a file to get the load and current position 
# 
#
#following the read write file for setup
import os
import matplotlib.pyplot as plt
import time
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
        
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from scservo_sdk import *                    # Uses SCServo SDK library

# Control table address
ADDR_SCS_TORQUE_ENABLE     = 40
ADDR_SCS_GOAL_ACC          = 41
ADDR_SCS_GOAL_POSITION     = 42
ADDR_SCS_GOAL_SPEED        = 46
ADDR_SCS_PRESENT_POSITION  = 56
ADDR_SCS_LOAD = 60 # The current load address added


# Default setting
SCS_ID                      = 14                 # SCServo ID : 1
BAUDRATE                    = 1000000           # SCServo default baudrate : 1000000
DEVICENAME                  = '/dev/ttyUSB1'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

SCS_MINIMUM_POSITION_VALUE  = 800       # SCServo will rotate between this value
SCS_MAXIMUM_POSITION_VALUE  = 601       # and this value (note that the SCServo would not move when the position value is out of movable range. Check e-manual about the range of the SCServo you use.)
SCS_MOVING_STATUS_THRESHOLD = 20          # SCServo moving status threshold
SCS_MOVING_SPEED            = 1           # SCServo moving speed
SCS_MOVING_ACC              = 1           # SCServo moving acc
protocol_end                = 1            # SCServo bit end(STS/SMS=0, SCS=1)

index = 0
scs_goal_position = [SCS_MINIMUM_POSITION_VALUE, SCS_MAXIMUM_POSITION_VALUE] 

portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Get methods and members of Protocol
packetHandler = PacketHandler(protocol_end)
x_axis = []
y_axis = []

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

scs_comm_result, scs_error = packetHandler.write1ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_ACC, SCS_MOVING_ACC)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))

# Write SCServo acc
scs_comm_result, scs_error = packetHandler.write1ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_ACC, SCS_MOVING_ACC)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))

# Write SCServo speed
scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_SPEED, SCS_MOVING_SPEED)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))

def get_feedback():
    data_read, results, error = packetHandler.readTxRx(portHandler, SCS_ID, ADDR_SCS_PRESENT_POSITION, 15)
    if len(data_read) ==  15:
        state = {
            'time': time.time(), # Time of feedback capture
            'position': SCS_MAKEWORD(data_read[0], data_read[1]),
            'speed':  SCS_TOHOST(SCS_MAKEWORD(data_read[2], data_read[3]),15),
            'load': SCS_MAKEWORD(data_read[4], data_read[5])/1000.0,
            'voltage': data_read[6]/10.0,
            'temperature': data_read[7],
            'status': data_read[9],
            'moving': data_read[10],
            'current': SCS_MAKEWORD(data_read[13], data_read[14]),
            }
    return state,results,error
scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_POSITION, 400)
time.sleep(0.1)
state,__,__ = get_feedback()
print(state["position"])
portHandler.closePort()
time.sleep(5)
portHandler.openPort()
state,__,__ = get_feedback()
print(state["position"])

    
    


if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))
portHandler.closePort()
plt.plot(x_axis, y_axis)
plt.xlabel('position')
plt.ylabel('load')
plt.show()


print("Press any key to terminate...")
getch()	
quit()







