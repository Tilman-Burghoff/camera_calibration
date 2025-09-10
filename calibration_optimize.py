# TODO think through this to figure out which transformation is needed

# This script computes the camera frame by solving the optimization problem
# min_P ||Y - XP^T|| where X contains rows (x,y,z,1) with the marker positions in the
# camera frame in homogeneous coordinates and Y the corresponding positions (x,y,z)
# in the gripper frame. The resulting matrix P is the transformation from gripper to camera frame.
# It has the closed form solution P = Y^T X (X^T X)^{-1} and is in the form P = [R|t].
# From there we can extract the rotation as a quaternion q and the translation vector t
# to get the needed frame transformation Q=(t,q)

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

df = pd.read_csv("calibration_data.csv", header=0)
print(df, df.columns)
X = df[['cam_x', 'cam_y', 'cam_z']].to_numpy()
X = np.hstack([X, np.ones((X.shape[0],1))])
Y = df[['gt_x', 'gt_y', 'gt_z']].to_numpy()

P = Y.T @ X @ np.linalg.inv(X.T @ X)
R_mat = P[:, :3]
t = P[:, 3]
q = R.from_matrix(R_mat).as_quat()

print('Calibration vector (t, q):')
print(t, q)