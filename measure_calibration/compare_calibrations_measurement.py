import numpy as np
import robotic as ry
from robotic.src.h5_helper import H5Reader
import matplotlib.pyplot as plt
from cv2 import aruco
import pickle
from pathlib import Path

def look_at_marker_ik(C, marker_name, distance=0.2):
    komo = ry.KOMO(C, 1, 1, 0, True)
    komo.addControlObjective([], 0), 1e-1
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.positionDiff, ['l_cameraWrist_o', marker_name], ry.OT.eq, np.eye(3), [0,0,distance])
    komo.addObjective([], ry.FS.scalarProductXZ, ['origin', 'l_cameraWrist_o'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYZ, ['origin', 'l_cameraWrist_o'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductZZ, ['origin', 'l_cameraWrist_o'], ry.OT.ineq)
    komo.addObjective([], ry.FS.scalarProductXY, ['l_cameraWrist_o', 'origin'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYY, ['origin', 'l_cameraWrist_o'], ry.OT.ineq)

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
C.addFrame('aruco_calibrated_camera_old', 'l_panda_joint7').setRelativePose([-0.021096, 0.0529521, 0.174296, 0.386088, 0.00567668, -0.0017275, -0.922443])
C.addFrame('aruco_calibrated_camera_new', 'l_panda_joint7').setRelativePose([-0.00499173, 0.066226, -0.0274848, 0.385713, 0.0288768, 0.0163762, -0.922021])

root = Path(__file__).parent.parent
h5 = H5Reader(root / 'data/marker_gt_fixed.h5')
manifest = h5.read_dict('manifest')

gt_pos = []
for id in manifest['marker_ids']:
    position = h5.read(f'marker_{id}/position')
    C.addFrame(f'marker_{id}', 'l_panda_base').setRelativePosition(position)
    gt_pos.append(position)
gt_pos = np.array(gt_pos)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)

bot = ry.BotOp(C, True)
bot.getImageAndDepth('l_cameraWrist') # initialize camera

results = [dict(), dict(), dict()]

for i in range(3):
    for id in manifest['marker_ids']:
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
        cam_position = np.average(corner_points, axis=0)
        for j, cam in enumerate(['table_calibrated_camera', 'aruco_calibrated_camera_old', 'aruco_calibrated_camera_new']):
            base_coords = cam_to_base(C, cam, cam_position)
            results[j].setdefault(id, []).append(base_coords[:3])

with open(root / 'data/comparison_results_axis_aligned.pkl', 'xb') as f:      
    pickle.dump(results, f)
