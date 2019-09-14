import scipy.io as sio
import numpy as np
import math
Obj = sio.loadmat('GaitLibrary_PyBullet2.mat')['GaitLibrary']
Velocity    = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FrontStance = Obj[0][0][1][0]
BackStance  = Obj[0][0][2][0]
FrontStance_LegPose = FrontStance[0][0]         # 6 velocities by 3 poses by 21 nodes
FrontStance_MotorV = FrontStance[0][1]      # 6 velocities by 2 motors by 21 nodes
FrontStance_Time = FrontStance[0][2]            # 6 velocities by 21 nodes
BackStance_LegPose = BackStance[0][0]           # 6 velocities by 3 poses by 21 nodes
BackStance_MotorV = BackStance[0][1]        # 6 velocities by 2 motors by 21 nodes
BackStance_Time = BackStance[0][2]              # 6 velocities by 21 nodes

def clamp(current_velocity, reference_velocity):
    lower_velocity = math.floor(current_velocity * 10) / 10
    upper_velocity = math.ceil(current_velocity * 10) / 10
    lower_percentage = 1 - (current_velocity - lower_velocity) / 0.1
    lower_index = reference_velocity.index(lower_velocity)
    upper_index = reference_velocity.index(upper_velocity)
    return lower_velocity, upper_velocity, lower_percentage, lower_index, upper_index

def interp2D(current_velocity, reference_velocity, data_2D):
    lower_velocity, upper_velocity, lower_percentage, lower_index, upper_index = clamp(current_velocity, reference_velocity)
    data_1D = lower_percentage * data_2D[lower_index] + (1 - lower_percentage) * data_2D[upper_index]
    return data_1D

def interp3D(current_velocity, reference_velocity, data_3D):
    lower_velocity, upper_velocity, lower_percentage, lower_index, upper_index = clamp(current_velocity, reference_velocity)
    data_2D = lower_percentage * data_3D[lower_index] + (1 - lower_percentage) * data_3D[upper_index]
    return data_2D

desired_LegPose = interp3D(0.49, Velocity, FrontStance_LegPose) # 3 pose by 21 nodes
desired_MotorV = interp3D(0.49, Velocity, FrontStance_MotorV)   # 3 pose by 21 nodes
desired_Time = interp2D(0.49, Velocity, FrontStance_Time)       # 21 nodes

desired_extension = np.reshape(desired_LegPose[2], len(desired_LegPose[2]))
desired_t = np.reshape(desired_Time, len(desired_Time[0]))
print(np.interp(0.24, desired_t, desired_extension))
