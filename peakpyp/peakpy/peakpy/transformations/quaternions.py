import numpy as np

def rotation_matrix_to_quaternion_numpy(matrix):
    m = matrix
    t = np.trace(m)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S

    ### not the faintest idea why do we put the minus sign here, but the result agrees with D2S2 and astropy
    ## i think it is just for the fact that q, and -q represent the same rottion
    return -np.array([qx, qy, qz, qw])

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    
    Parameters:
        q (array-like): Quaternion [qw, qx, qy, qz].
        
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    qx, qy, qz, qw = q

    # Compute the components of the rotation matrix
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    
    return R



def quaternion_multiply(quaternion1, quaternion2):
    b1, c1, d1, a1 = quaternion1
    b2, c2, d2, a2 = quaternion2

    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d = a1*d2 + b1*c2 - c1*b2 + d1*a2
    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    return np.array([b,c,d,a])

def quaternion_conjugate(quaternion):
    x, y, z, w = quaternion
    return np.array([-x, -y, -z, w])

def rotate_quaternion(vector, rotation_quaternion):
    quaternion = np.concatenate([vector, [0]])
    return quaternion_multiply(rotation_quaternion, quaternion_multiply(quaternion, quaternion_conjugate(rotation_quaternion))) 