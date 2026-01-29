import numpy as np
## def 3d rotation tables
def Rx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    return np.round(np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]]), 8)


def Rzyx(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    # print('Rx:', Rx(roll))
    # print('Ry:', Ry(pitch))
    # print('Rz:', Rz(yaw))

    return np.round(Rx(roll)@ (Ry(pitch)@Rz(yaw)), 5)

def Rxyz(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    # print('Rx:', np.round(Rx(roll), 2))
    # print('Ry:', np.round(Ry(pitch), 2))
    # print('Rz:', np.round(Rz(yaw), 2))
    return np.round(Rz(yaw)@ Ry(pitch)@Rx(roll), 5)

def Rzyx_clockwise(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    return np.round(Rx(roll).T @ (Ry(pitch).T @ Rz(yaw).T, 2), 5)

def Rzxy(roll, pitch, yaw):

    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    return Ry(pitch) @ (Rx(roll) @ Rz(yaw))




def dcm2euler_zyx_clockwise(C):
    # Extract pitch (theta)
    theta = np.arcsin(-C[0, 2])

    if np.abs(C[0, 2]) < 1:  # No gimbal lock
        psi = np.arctan2(C[0, 1], C[0, 0])  # Yaw
        phi = np.arctan2(C[1, 2], C[2, 2])  # Roll
    else:  # Gimbal lock
        print('RED ALERT, GIMBAL LOCK IS DETECTED')
        # psi = np.arctan2(-C[1, 0], C[1, 1])  # Yaw
        # phi = 0  # Set roll to zero
        psi = 0
        phi = np.arctan2(C[1, 0], C[1, 1])
    return np.degrees(phi), np.degrees(theta), np.degrees(psi) 




def dcm2euler_xyz(C):
    # Extract pitch (theta)
    pitch = np.arcsin(-C[2, 0])
    print(np.degrees(pitch))

    if np.abs(C[2,0]) < 1:  # No gimbal lock
        roll = np.arctan2(C[2, 1], C[2, 2])  
        yaw = np.arctan2(C[1, 0], C[0, 0])  
    else:  # Gimbal lock
        print('RED ALERT, GIMBAL LOCK IS DETECTED')
        yaw = 0
        roll = np.arctan2(-C[1, 2], C[1, 1])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw) 

### careful you are in dangerous waters - what did i do here?
def dcm2euler_xyz(C):
    # Extract pitch (theta)
    pitch = np.arcsin(-C[2, 0])
    print(np.degrees(pitch))

    if np.abs(C[2,0]) < 1:  # No gimbal lock
        roll = np.arctan2(C[2, 1], C[2, 2])  
        yaw = np.arctan2(C[1, 0], C[0, 0])  
    else:  # Gimbal lock
        print('RED ALERT, GIMBAL LOCK IS DETECTED')
        roll = 0
        yaw = np.arctan2(-C[0, 1], C[1, 1])  
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw) 




#################### DEPRECATED #########################
def dcm2euler_zyx_orthogonal(dcm, frot = Rzyx_clockwise):
    """
    If it uses a clockwise or a counterclockwise will depend on the function frot
    
    
    """
    groups = 0 ## added because i did not like seeing the error. But it is deprecated, and should not be used, unless found necessary.
    return_angles = []
    for gg in groups:
        if np.array_equal(np.round(frot(gg[0], gg[1], gg[2]), 2), np.round(dcm, 2)):
            return_angles.append(gg)
    return return_angles
