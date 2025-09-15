# we parse our data into the form (n_i, c_i, x_i), where n_i is the normal of the table and c_i its center expressed in the gripper frame, while
# x_i is a point of the pointcloud in the the camera frame in homogeneous coordinates. We try to find the projection P from camera to gripper
# such that all points lie on the table, ie n_i^T * (P * x_i - c_i) = 0 for all i. We will try to solve this by linearization and by pseudo-inverse
# (see respective functions below). The resulting P is a 3x4 matrix that combines a rotation R and translation t in the form P = [R | t],
# from which we first recover R and t and then our transformation Q = [t, q] where q is the quaternion describing R.

import json
import numpy as np
import robotic as ry
from scipy.spatial.transform import Rotation


SOLVER = 'lin' # 'lin' for linearization, 'pinv' for pseudo-inverse


def load_data(C, filename):
    with open(filename, "r") as f:
        data = json.load(f)

    n_all = []
    c_all = []
    x_all = []
    pcls = []
    qs = []

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
        pcls.append(points)
        qs.append(entry['joint_state'])

    n_all = np.vstack(n_all)
    c_all = np.vstack(c_all)
    x_all = np.vstack(x_all)

    return n_all, c_all, x_all, pcls, qs


def solve_by_linerization(n, c, x):
    # We rewrite the problem as n_i^T * c_i = n_i^T * P * x_i = kron(n_i, x_i) * vec(P), 
    # which allows us to solve for vec(P) with least squares.
    A = np.einsum('ij,ik->ijk', n, x).reshape(-1, 12) # each row is kron(n_i, x_i)
    b = np.einsum('ij,ij->i', n, c) # each row is n_i^T * c_i
    P,_,_,_ = np.linalg.lstsq(A, b)
    return P.reshape(3,4)


def solve_by_pinv(n, c, x):
    # We use N P X^T = N C.T => P = N^+ N C (X^T)^+ where N, X, C are the matrices with rows n_i, x_i, c_i
    P = np.linalg.pinv(n) @ n @ c.T @ np.linalg.pinv(x.T)
    return P


def solve_by_closed_form(n, c, x):
    # we attempt to find a closed form solution by deriving ||n_i^T * (P * x_i - c_i)||^2 wrt P.
    # This leads to the equation sum_i (n_i^T P x_i) (n_i x_i^T) = sum_i (n_i^T c_i) n_i x_i^T = A
    # Let S = [vec(n_i x_i^T)]_i be the matrix with rows vec(n_i x_i^T) and solve for s in S s = vec(A)
    # Now we recover P by S vec(P) = s. This leads to the closed form vec(P) = S(S^T S)^-1 S^-1 vec(A)
    A = np.einsum('ij, ij, ik, il->kl', n, c, n, x)
    S = np.einsum('ij, ik->ijk', n, x).reshape(-1, 12)
    s,_,_,_ = np.linalg.lstsq(S.T, A.ravel(), rcond=None)
    print(S.shape, A.shape, s.shape)
    P,_,_,_ = np.linalg.lstsq(S, s, rcond=None)
    return P.reshape(3,4)



def get_Q(P):
    R = P[:3, :3].T
    t = -R @ P[:3, 3]
    q = Rotation.from_matrix(R).as_quat()
    Q = np.round(t, 8).tolist() + np.round(q, 8).tolist()
    return Q


def visualize_Q(C, Q, pcls, qs):
    cam = C.addFrame('computed_cam', 'l_gripper').setShape(ry.ST.marker, [.1])
    cam.setRelativePose(Q)

    pcl_f = C.addFrame('pcl', 'computed_cam')
    for i, (pcl, q) in enumerate(zip(pcls, qs)):
        pcl_f.setPointCloud(pcl)
        C.setJointState(q)
        C.view(True, f'pointcloud {i} in computed camera frame')


def main():
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))
    
    n, c, x, pcls, qs = load_data(C, 'camera_calibration_data.json')
    #if SOLVER == 'lin':
    #    P = solve_by_linerization(n, c, x)
    #elif SOLVER == 'pinv':
    #    P = solve_by_pinv(n, c, x)
    #else:
    #    raise ValueError("Invalid SOLVER option")
    P = solve_by_closed_form(n, c, x)
    # both methods produce different, incorrect results

    Q = get_Q(P)
    print(Q) # returns [0.02063466, 0.05156388, 0.03764117, 0.00366712, 0.99988991, -0.00660188, 0.0127722] for lin
    # TODO: weirdly mostly correct, but the y translation is in the wrong direction and the rotation should 
    # align y instead of x axis with the gripper.

    visualize_Q(C, Q, pcls, qs)


if __name__ == '__main__':
    main()