# This script gathers calibration data.
# sample n number of points (arUco marker centers) from fixed camera position set by user
# find ground truth positions by moving the gripper to the point, use gripper frame.

from aruco_tracker import Smoothing, ArucoTracker
import robotic as ry
import numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation as R
from time import sleep

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle_camera.g'))
bot = ry.BotOp(C, True)
tracker = ArucoTracker(C)
rgb, _, points = bot.getImageDepthPcl('l_cameraWrist')
pcl = C.addFrame('pcl', 'l_cameraWrist')
pcl.setPointCloud(points, rgb)

iters = int(input("Number of iterations:"))
visited = set()

no_datapoints = 0
measurements = []
for i in range(iters):
    tracker.position_smoothing = Smoothing.LAST
    tracker.reset_average()
    print("Move bot into position, press q when done")
    bot.hold(floating=True, damping=False)
    while bot.getKeyPressed() != ord('q'):
        rgb, depth, points = bot.getImageDepthPcl('l_cameraWrist')
        pcl.setPointCloud(points, rgb)
        visible = tracker.track_markers(rgb, points)
        tracker.tag_markers()
        bot.sync(C, viewMsg=f"Move bot such that markers are visible, then press q.\n visible Markers: {visible}")

    bot.hold(floating=False, damping=True)
    C.view(False)
    q = C.getJointState()
    print("Collecting measurements")
    for i in range(10):
        rgb, depth, points = bot.getImageDepthPcl('l_cameraWrist')
        visible = tracker.track_markers(rgb, points)
        tracker.tag_markers()
        pcl.setPointCloud(points, rgb)
        bot.sync(C, .1, "Collecting measurements")
        tracker.position_smoothing = Smoothing.AVERAGE

    visited.update(set(visible))
    measurements.append((q, {id: tracker[id] for id in visible}))
    no_datapoints += len(visible)


print("Gathering ground truth")
ground_truth = {}
print(visited)
for id in list(visited):
    print(f"Move gripper to marker {id}, then press q")
    bot.hold(floating=True, damping=False)

    while bot.getKeyPressed() != ord('q'):
        bot.sync(C, viewMsg=f"Move gripper to marker {id}, then press q")

    ground_truth[id] = C.eval(ry.FS.position, ['l_gripper'])[0]
    C.view(True)
    bot.sync(C)

bot.home(C)
bot.wait(C)
del bot

with open('calibration_data.csv', 'x') as f:
    f.write("cam_x, cam_y, cam_z, gt_x, gt_y, gt_z\n")
    for q, measurement in measurements:
        C.setJointState(q)
        gripper_transform = np.linalg.inv(C.getFrame('l_gripper').getTransform())
        for id, z_cam in measurement.items():
            z_true = gripper_transform @ np.concatenate((ground_truth[id], [1]))
            f.write(f"{z_cam[0]}, {z_cam[1]}, {z_cam[2]}, {z_true[0]}, {z_true[1]}, {z_true[2]}\n")