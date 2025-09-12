import robotic as ry
import numpy as np
import json

NUMBER_OF_POSES = 20
MIN_ANGLE = 0
MAX_ANGLE = 5/6 * np.pi
FILTER_PCL_MIN_DISTANCE = 0.2
FILTER_PCL_MAX_DISTANCE = 0.8

C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))
bot = ry.BotOp(C, True)
pcl = C.addFrame("pcl", "l_cameraWrist")
bot.getImageAndDepth('l_cameraWrist') # initialize camera

data = []

for i in range(NUMBER_OF_POSES):
    bot.hold(floating=True, damping=False)
    while bot.getKeyPressed() != ord('q'):
        rgb, _, points = bot.getImageDepthPcl("l_cameraWrist")
        points = points.reshape(-1,3)
        rgb = rgb.reshape(-1,3)
        mask = (np.linalg.norm(points, axis=-1) > FILTER_PCL_MIN_DISTANCE) & (np.linalg.norm(points, axis=-1) < FILTER_PCL_MAX_DISTANCE)
        pcl.setPointCloud(points[mask], rgb[mask])
        bot.sync(C, viewMsg="move to a new pose and press 'q' to capture")
    
    bot.hold(floating=False)
    bot.sync(C)
    _,_, points = bot.getImageDepthPcl("l_cameraWrist")
    points = points.reshape(-1,3)
    mask = (np.linalg.norm(points, axis=-1) > FILTER_PCL_MIN_DISTANCE) & (np.linalg.norm(points, axis=-1) < FILTER_PCL_MAX_DISTANCE)
    data.append({
        'id': i,
        'gripper_pose': C.getFrame("l_gripper").getPose().tolist(), 
        'joint_state':C.getJointState().tolist(),
        'pointcloud': points[mask].tolist()
    })
    

del bot
del C

with open("camera_calibration_data.json", "w") as f:
    json.dump(data, f, indent=2)