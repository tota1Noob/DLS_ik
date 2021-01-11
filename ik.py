import numpy as np
import math
import time
from scipy.spatial.transform import Rotation as R
np.set_printoptions(suppress=True)

def calculate_t(theta, d, a, alpha):
    x = np.zeros([4,4], dtype = float) 
    x[0, 0] = math.cos(theta)
    x[0, 1] = -math.sin(theta) * math.cos(alpha)
    x[0, 2] = math.sin(theta) * math.sin(alpha)
    x[0, 3] = a * math.cos(theta)

    x[1, 0] = math.sin(theta)
    x[1, 1] = math.cos(theta) * math.cos(alpha)
    x[1, 2] = -math.cos(theta) * math.sin(alpha)
    x[1, 3] = a * math.sin(theta)

    x[2, 1] = math.sin(alpha)
    x[2, 2] = math.cos(alpha)
    x[2, 3] = d
    x[3, 3] = 1

    return x

def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

def forward_kinematics(dh, joints):
    row = dh.shape[0]
    joint_len = len(joints)
    trans = np.zeros([row, 4, 4], dtype = float)
    for i in range(row):
        theta = dh[i, 0]
        if i < joint_len:
            theta = theta + math.radians(joints[i])
        trans[i] = calculate_t(theta, dh[i, 1], dh[i, 2], dh[i, 3])

    d0_n = np.identity(4)
    for i in range(row):
        d0_n = d0_n.dot(trans[i])

    euler = rotationMatrixToEulerAngles(d0_n[:3, :3])

    posNrot = [round(d0_n[0, 3], 2),
               round(d0_n[1, 3]),
               round(d0_n[2, 3]),
               round(euler[0]),
               round(euler[1]),
               round(euler[2])]
    return d0_n, euler, posNrot

def calculate_jacobian(dh, joints):
    row = dh.shape[0]
    joint_len = len(joints)
    trans = np.zeros([row, 4, 4], dtype = float)
    for i in range(row):
        theta = dh[i, 0]
        if i < joint_len:
            theta = theta + math.radians(joints[i])
        trans[i] = calculate_t(theta, dh[i, 1], dh[i, 2], dh[i, 3])

    d0_n = np.identity(4)
    for i in range(row):
        d0_n = d0_n.dot(trans[i])

    #print(d0_n)
    d0_n = d0_n[:3, 3]

    jacobian = np.zeros([6,row], dtype = float)
    #print(jacobian.shape)
    for i in range(row):
        current_trans = np.identity(4)
        for j in range(i):
            current_trans = current_trans.dot(trans[j])
        lower_three = current_trans[:3, 2]
        d = current_trans[:3, 3]
        upper_three = np.cross(lower_three, (d0_n - d))
        jacobian[0:3, i] = upper_three
        jacobian[3:6, i] = lower_three

    return jacobian

def inverse_kinematics(dh, current_joints, target_pos):
    dof = len(current_joints)
    _, _, current_forward = forward_kinematics(dh, current_joints)
    target_forward = target_pos
    error = [target_forward[i] - current_forward[i] for i in range(6)]
    print("error", error)
    err = np.linalg.norm(error)
    dampping = 10
    iter = 0
    maxIter = 10000
    while err > 0.0001:
        time_start = time.time()
        jacobian = calculate_jacobian(dh, current_joints)
        delta_theta = np.matmul(
                        np.matmul(
                            jacobian.T, 
                            np.linalg.inv(
                                np.add(
                                    np.matmul(
                                        jacobian,
                                        jacobian.T
                                    ),
                                    dampping * dampping * np.identity(dof)
                                )
                            )
                        ),
                        error
                    )
        if iter == 0:
            print(delta_theta)
        current_joints = [delta_theta[i] + current_joints[i] for i in range(dof)]
        time_end = time.time()
        #print(f'iter {iter} took: ', time_end - time_start)

        iter += 1
        if(iter > maxIter):
            break
        if(np.linalg.norm(delta_theta)) < 0.01:
            break
        
        _, _, current_forward = forward_kinematics(dh, current_joints)
        error = [target_forward[i] - current_forward[i] for i in range(6)]
        err = np.linalg.norm(error)

    #print("Position based on ik: ", current_forward)
    return current_joints


#dh parameters in the order of [theta, d, a, alpha]
dh = np.zeros([6, 4], dtype = float)
dh[0,:] = [0, 345, 50, math.radians(90)]
dh[1,:] = [0, 0, 420, 0]
dh[2,:] = [0, 0, 45, math.radians(90)]
dh[3,:] = [0, 440, 0, math.radians(-90)]
dh[4,:] = [0, 0, 0, math.radians(90)]
dh[5,:] = [0, 73 + 205, 0, math.radians(180)]

joints = [60.55, 36.86, 4.77, 0, -41.63, 30.56]
_, _, result = forward_kinematics(dh, joints)
print(result)


current_joints = [30., 90., 0., 0., -90., 0.]
target_pos = [350.0, 620.0, 20.0, 0.0, 0.0, 30.0]

#do inverse kinematics and time it
time_start=time.time()
calculated_joints = inverse_kinematics(dh, current_joints, target_pos)
time_end=time.time()
print('Total time cost: ',time_end - time_start)
print("ik joint values: ", calculated_joints)