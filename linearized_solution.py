# we assume data in the form (n_i, c_i, x_i), where n_i is the normal of the table and c_i its center expressed in the gripper frame.
# x_i is a point of the pointcloud in the the camera frame in homogeneous coordinates. We try to find the projection P from camera to gripper
# such that all points lie on the table, ie n_i^T * (P * x_i - c_i) = 0 for all i. This can be rewritten as a linear system of equations by
#  n_i^T * (P * x_i - c_i) = 0  <=> n_i^T * c_i = n_i^T * T * x_i = kron(n_i, x_i) * vec(P), which allows us to solve for vec(P) with least squares.
# P then has the form P = [R^T, -R^T * t], from which we first recover R and t and then our transformation Q = [t, q] where q is the quaternion of R.

import json
import numpy as np
import robotic as ry
from scipy.spatial.transform import Rotation


C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))
qHome = C.getJointState()

def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)


    n_all = []
    c_all = []
    x_all = []

    for entry in data:
        C.setJointState(entry['joint_state'])
        c, _ = C.eval(ry.FS.positionRel, ['origin', 'l_gripper'])
        n, _ = C.eval(ry.FS.vectorZRel, ['world','l_gripper']) # the table normal is the global z vector

        points = np.array(entry['pointcloud'])
        ones = np.ones((points.shape[0], 1))
        points_hom = np.hstack([points, ones])  # convert to homogeneous coordinates

        n_all.append(np.repeat(n[None, :], points.shape[0], axis=0))
        c_all.append(np.repeat(c[None, :], points.shape[0], axis=0))
        x_all.append(points_hom)

    n_all = np.vstack(n_all)
    c_all = np.vstack(c_all)
    x_all = np.vstack(x_all)

    return n_all, c_all, x_all

if __name__ == '__main__':
    n, c, x = load_data('camera_calibration_data.json')
    A = np.einsum('ij,ik->ijk', n, x).reshape(-1, 12) # each row is kron(n_i, x_i)
    b = np.einsum('ij,ij->i', n, c) # each row is n_i^T * c_i
    P,_,_,_ = np.linalg.lstsq(A, b)
    P = P.reshape(3,4)
    R = P[:3, :3].T
    t = -R @ P[:3, 3]
    q = Rotation.from_matrix(R).as_quat()
    Q = np.round(t, 8).tolist() + np.round(q, 8).tolist()
    print(Q) # returns [0.02063466, 0.05156388, 0.03764117, 0.00366712, 0.99988991, -0.00660188, 0.0127722]
    # TODO: weirdly mostly correct, but the y translation is in the wrong direction and the rotation should 
    # align y instead of x axis with the gripper. This might be due to the image coordinate frame?? 
    # but since we use the pointcloud this shouldnt matter...

    C.setJointState(qHome)
    cam = C.addFrame('computed_cam', 'l_gripper').setShape(ry.ST.marker, [.1])
    cam.setRelativePose(Q)
    C.view(True, 'computed camera pose')
