# This script computes the camera frame by solving the optimization problem
# argmin_theta 1/n sum_(z_cam, z_true) ||T_theta * z_true - z_cam|| st. ||q|| = 1
# using the data (z_cam, z_true) collected by calibration_data_collection.py
# theta = (t, q)\in R^7 is the transformation vector consisting of the translation t and rotation q as a normalized quaternion.
# T is the homogenous transformation matrix T = [[R_q,t], [0,1]], where R_q is the rotation corresponding to the quaternion

def objective(x):
    t = x[:3]
    q = x[3:]
    transform = np.zeros((4,4))
    transform[:3, :3] = R.from_quat(q).as_matrix()
    transform[:3, 3] = t
    transform[3, 3] = 1.

    error = 0
    for q, measurement in measurements:
        C.setJointState(q)
        gripper_transform = np.linalg.inv(C.getFrame('l_gripper').getTransform())
        for id, z_cam in measurement.items():
            z_true = gripper_transform @ np.concatenate((ground_truth[id], [1]))
            error += np.linalg.norm((transform @ z_true)[:3] - z_cam)**2/2

    return error/no_datapoints


def get_dRdq(q):
    # because im lazy
    q0, q1, q2, q3 = q
    dRdq0 = np.array([
        [4*q0, -2*q3, 2*q2],
        [2*q3, 4*q0, -2*q1],
        [-2*q2, 2*q1, 4*q0]
    ])
    dRdq1 = np.array([
        [4*q1, 2*q2, 2*q3],
        [2*q2, 0, -2*q0],
        [2*q3, 2*q0,0]
    ])
    dRdq2 = np.array([
        [0, 2*q1, 2*q0],
        [2*q1, 4*q2, 2*q3],
        [-2*q0, 2*q3, 0]
    ])

    dRdq3 = np.array([
        [0, -2*q0, 2*q1],
        [2*q0, 0, 2*q2],
        [2*q1, 2*q2, 4*q3]
    ])

    return np.stack([dRdq0, dRdq1, dRdq2, dRdq3], axis=0)


def objective_grad(x):
    t = x[:3]
    q = x[3:]
    transform = np.zeros((4,4))
    transform[:3, :3] = R.from_quat(q).as_matrix()
    transform[:3, 3] = t
    transform[3, 3] = 1.

    grad = np.zeros(7) # grad =(t,q)
    dRdq = get_dRdq(q) # matrix of shape (4,3,3) with dRdq[i] = dR/dq_i
    for q, measurement in measurements:
        C.setJointState(q)
        gripper_transform = np.linalg.inv(C.getFrame('l_gripper').getTransform())
        for id, z_cam in measurement.items():
            z_true = gripper_transform @ np.concatenate((ground_truth[id], [1]))
            inner = ((transform @ z_true)[:3] - z_cam)
            grad[:3] += inner
            grad[3:] += np.dot(np.dot(dRdq, z_true), inner) # grad_q = inner_j * dRdq_ijk * z_true_k
    return grad


def constraint_val(x):
    q = x[3:]
    return 1/2*(np.linalg.norm(q)**2 - 1)

def constraint_grad(x):
    q = x[3:]
    grad = np.zeros(7)
    grad[3:] = q
    return grad[None, :] # scipy expects 1x7 jacobian 


constraint = optimize.NonlinearConstraint(constraint_val, 0,0, jac=constraint_grad)
ret = optimize.minimize(
    objective, 
    jac=objective_grad,
    x0=np.array([0,0,0,1,0,0,0]),
    constraints=constraint
)

print(ret)