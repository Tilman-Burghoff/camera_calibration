from cv2 import aruco
import robotic as ry
import numpy as np
from robotic.src.h5_helper import H5Writer, H5Reader
import json

# TODO: Ground truth

aruco_6x6 = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters_create()
aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX


NUMBER_OF_POSES = 20
MIN_DISTANCE = 0.2
MAX_DISTANCE = 0.8

C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))
bot = ry.BotOp(C, True)
pcl = C.addFrame("pcl", "l_cameraWrist")
bot.getImageAndDepth('l_cameraWrist') # initialize camera

h5 = H5Writer('aruco_calibration_data.h5')

markers = set()
i = 0
while i < NUMBER_OF_POSES:
    bot.hold(floating=True, damping=False)
    while bot.getKeyPressed() != ord('q'):
        rgb, _, points = bot.getImageDepthPcl("l_cameraWrist")
        corners, ids, _ = aruco.detectMarkers(rgb, aruco_6x6)
        points = points.reshape(-1,3)
        rgb = rgb.reshape(-1,3)
        mask = (np.linalg.norm(points, axis=-1) > MIN_DISTANCE) & (np.linalg.norm(points, axis=-1) < MAX_DISTANCE)
        pcl.setPointCloud(points[mask], rgb[mask])
        if ids is not None:
            viewmsg = f"visible markers: {ids.squeeze().tolist()}\nmove to a new pose and press 'q' to capture"
        else:
            viewmsg = f"no markers visible\nmove to a new pose and press 'q' to capture"
        bot.sync(C, viewMsg=viewmsg)
    
    bot.hold(floating=False)
    bot.sync(C)
    joint_state = C.getJointState()
    rgb, _, points = bot.getImageDepthPcl("l_cameraWrist")
    corners, ids, _ = aruco.detectMarkers(rgb, aruco_6x6, parameters=aruco_params)

    if ids is None:
        print("No markers detected, please try again")
        continue
    
    marker_set = set(ids.squeeze().tolist()) if ids.shape[0] > 1 else {int(ids)} # maybe alt set(ids.flatten().tolist())
    print(marker_set)
    markers.update(marker_set)
    
    centers = [np.mean(np.array(corners[i].squeeze()), axis=0) for i in range(len(corners))]
    key = f'dataset_{i}'
    h5.write(key+'/joint_state', joint_state, dtype='float64')
    h5.write(key+'/centers', centers, dtype='float32')
    h5.write(key+'/ids', ids.flatten(), dtype='int32')
    i += 1

del bot
del C

print(f"Recorded {i} poses with markers: {markers}")

manifest = {
    'description': 'for various poses: joint state of the panda and ids and centers of arUco markers in pixel coordinates',
    'n_datasets': NUMBER_OF_POSES,
    'marker_ids': list(markers),
    'keys': ['manifest', 'dataset_[i]/joint_state', 'dataset_[i]/centers', 'dataset_[i]/ids']
}
h5.write('manifest', bytearray(json.dumps(manifest), 'utf-8'), dtype='int8')