from cv2 import aruco
import robotic as ry
import numpy as np
import json

# TODO: Ground truth

aruco_6x6 = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX


NUMBER_OF_POSES = 20
MIN_DISTANCE = 0.2
MAX_DISTANCE = 0.8

C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))
bot = ry.BotOp(C, True)
pcl = C.addFrame("pcl", "l_cameraWrist")
bot.getImageAndDepth('l_cameraWrist') # initialize camera

data = []

while i < NUMBER_OF_POSES:
    bot.hold(floating=True, damping=False)
    while bot.getKeyPressed() != ord('q'):
        rgb, _, points = bot.getImageDepthPcl("l_cameraWrist")
        points = points.reshape(-1,3)
        rgb = rgb.reshape(-1,3)
        mask = (np.linalg.norm(points, axis=-1) > MIN_DISTANCE) & (np.linalg.norm(points, axis=-1) < MAX_DISTANCE)
        pcl.setPointCloud(points[mask], rgb[mask])
        bot.sync(C, viewMsg="move to a new pose and press 'q' to capture")
    
    bot.hold(floating=False)
    bot.sync(C)
    rgb, _, points = bot.getImageDepthPcl("l_cameraWrist")
    corners, ids, _ = aruco.detectMarkers(rgb, aruco_6x6, parameters=aruco_params)

    if ids is None:
        print("No markers detected, please try again")
        continue

    markers = []
    for corner, id in zip(corners, ids.flatten()):
        corners = corner.squeeze()
        center = np.mean(corners, axis=0)
        markers.append({
            'id': id, 
            'corners': corner.squeeze().tolist(), 
            'center': center.tolist()
        })

    data.append({
        'id': i,
        'gripper_pose': C.getFrame("l_gripper").getPose().tolist(), 
        'joint_state':C.getJointState().tolist(),
        'markers': markers
    })
    i += 1


del bot
del C

with open("aruco_calibration_data.json", "w") as f:
    json.dump(data, f, indent=2)