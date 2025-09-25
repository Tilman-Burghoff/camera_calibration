import numpy as np
import robotic as ry
from robotic.src.h5_helper import H5Reader
import matplotlib.pyplot as plt
from cv2 import aruco
import pickle
from pathlib import Path
import random

def look_at_marker_ik(C, marker_name, distance=0.2):
    komo = ry.KOMO(C, 1, 1, 0, True)
    komo.addControlObjective([], 0), 1e-1
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.positionDiff, ['l_cameraWrist', marker_name], ry.OT.eq, np.eye(3), [0,0,distance])
    komo.addObjective([], ry.FS.scalarProductXZ, ['origin', 'l_cameraWrist'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYZ, ['origin', 'l_cameraWrist'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductZZ, ['origin', 'l_cameraWrist'], ry.OT.ineq)
    komo.addObjective([], ry.FS.scalarProductXY, ['l_cameraWrist', 'origin'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYY, ['origin', 'l_cameraWrist'], ry.OT.ineq)

    ret = ry.NLP_Solver(komo.nlp(), verbose=-1).solve()
    print(ret)
    return komo.getPath()[-1]

def invert_transform(T):
    T_inv = np.eye(4)
    R_inv = T[:3, :3].T
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -R_inv @ T[:3, 3]
    return T_inv

def cam_to_base(C: ry.Config, cam_frame, coords):
    coords = np.array([*coords, 1])
    cam = C.getFrame(cam_frame)
    base = C.getFrame('l_panda_base')
    return invert_transform(base.getTransform()) @ cam.getTransform() @ coords


C = ry.Config()
C.addFile(ry.raiPath('/scenarios/pandaSingle_camera.g'))
C.addFrame('table_calibrated_camera', 'l_panda_joint7').setRelativePose([-0.0209509, 0.0471966, 0.17127, 0.386655, 0.0133645, -0.00307343, -0.922122])
C.addFrame('aruco_calibrated_camera_20_poses', 'l_panda_joint7').setRelativePose([-0.0226106, 0.0513997, 0.169131, 0.389766, 0.00753701, -0.00502238, -0.920869])
C.addFrame('aruco_calibrated_camera_100_poses', 'l_panda_joint7').setRelativePose([-0.0245442, 0.0477194, 0.16876, 0.393552, 0.00961641, -0.00293342, -0.919247])

root = Path(__file__).parent.parent
h5 = H5Reader(root / 'data/marker_gt_new.h5')
manifest = h5.read_dict('manifest')

gt_pos = []
gt_opti_output = """optimal aruco_0 position: [0.0616964, 0.466489, -0.00147755]
optimal aruco_1 position: [0.0457536, -0.363669, -0.00128368]
optimal aruco_4 position: [0.558125, -0.128908, 0.00324898]
optimal aruco_6 position: [0.610304, 0.124817, 0.00249124]
optimal aruco_11 position: [0.355938, -0.307029, 0.000788678]
optimal aruco_12 position: [0.395202, 0.340062, -0.000933702]
optimal aruco_14 position: [0.38215, -0.0447043, 0.000646111]"""
for line in gt_opti_output.split('\n'):
    desc, pos_str = line.split(' position: ')
    id = int(desc.split('_')[1])
    marker = C.addFrame(f'marker_{id}', 'l_panda_base')
    marker.setShape(ry.ST.marker, [.05]).setColor([1,0,0,.5])
    pos = eval(pos_str)
    gt_pos.append(pos)
    marker.setRelativePosition(pos)
    print(f'marker {id} position: {pos}')
gt_pos = np.array(gt_pos)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)

bot = ry.BotOp(C, True)
bot.getImageAndDepth('l_cameraWrist') # initialize camera

results = [dict(), dict(), dict(), dict()]
markers = manifest['marker_ids'].copy()
for i in range(5):
    random.shuffle(markers)
    for id in markers:
        q = look_at_marker_ik(C, f'marker_{id}')
        bot.moveTo(q)
        bot.wait(C)

        rgb, _, pcl = bot.getImageDepthPcl('l_cameraWrist')
        corners, ids, _ = aruco.detectMarkers(rgb, aruco_dict)
        if ids is None or id not in ids:
            print(f'Marker {id} not detected!')
            continue
        idx = np.argmax(ids.flatten() == id)
        assert ids[idx] == id
        corners = corners[idx].squeeze().astype("int")
        corner_points = pcl[corners[:,1],corners[:,0]]
        cam_position = corner_points[0]
        for j, cam in enumerate(['table_calibrated_camera', 'aruco_calibrated_camera_20_poses', 'aruco_calibrated_camera_100_poses', 'l_cameraWrist_o']):
            base_coords = cam_to_base(C, cam, cam_position)
            results[j].setdefault(id, []).append(base_coords[:3])

with open(root / 'data/comparison_results_v3.pkl', 'xb') as f:      
    pickle.dump(results, f)
