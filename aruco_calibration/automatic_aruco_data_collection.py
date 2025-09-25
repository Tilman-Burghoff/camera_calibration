import robotic as ry
import numpy as np
import json
from robotic.src import h5_helper
from cv2 import aruco
from pathlib import Path
import matplotlib.pyplot as plt

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
aruco_params = aruco.DetectorParameters_create()
aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
root = Path(__file__).parent.parent

NUMBER_OF_POSES = 100
IMAGES_PER_POSE = 10
MIN_ANGLE = 0
MAX_ANGLE = 1/4 * np.pi
MIN_DISTANCE = .2
MAX_DISTANCE = .7
MAX_TARGET_OFFSET = 0.1
SEED = 0


def look_with_angle(C, target_name, distance, angle):
    komo = ry.KOMO(C, 1, 1, 0, True)
    
    komo.addControlObjective([], 0, 1e-1)

    height = distance * np.cos(angle)

    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1])
    komo.addObjective([], ry.FS.negDistance, ['l_cameraWrist', target_name], ry.OT.eq, [1], [-distance]) 
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    komo.addObjective([], ry.FS.positionDiff, ['l_cameraWrist', target_name], ry.OT.eq, [0,0,1], [0,0,height])
    # point camera at target
    komo.addObjective([], ry.FS.positionRel, [target_name, 'l_cameraWrist'], ry.OT.eq, [[1,0,0],[0,1,0]])

    return komo


def bilinear_depth_interpolation(depth, x, y):
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, depth.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, depth.shape[0] - 1)

    Q11 = depth[y0, x0]
    Q21 = depth[y0, x1]
    Q12 = depth[y1, x0]
    Q22 = depth[y1, x1]

    if np.isnan(Q11) or np.isnan(Q21) or np.isnan(Q12) or np.isnan(Q22):
        return np.nan

    return (Q11 * (x1 - x) * (y1 - y) +
            Q21 * (x - x0) * (y1 - y) +
            Q12 * (x1 - x) * (y - y0) +
            Q22 * (x - x0) * (y - y0))


def main():
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))

    h5 = h5_helper.H5Reader(root / 'data' / 'marker_gt_new.h5')
    manifest = h5.read_dict('manifest')

    for id in manifest['marker_ids']:
        key = f'marker_{id}/position'
        pos = h5.read(key)
        marker = C.addFrame(f'marker_{id}', 'l_panda_base')
        marker.setShape(ry.ST.marker, [.05]).setColor([1,0,0,.5])
        marker.setRelativePosition(pos.tolist() + [0])
        print(f'marker {id} position: {pos}')

    h5 = h5_helper.H5Writer(root / 'data' /'aruco_calibration_data_v2.h5')

    bot = ry.BotOp(C, True)
    bot.getImageAndDepth('l_cameraWrist') # initialize camera

    target = C.addFrame('target')

    rng = np.random.default_rng(SEED)

    marker_set = set()
    
    i = 0

    while i < NUMBER_OF_POSES:
        marker_id = rng.choice(manifest['marker_ids'])
        offset = rng.random(2) * MAX_TARGET_OFFSET
        target.setPosition(C.getFrame(f'marker_{marker_id}').getPosition() + np.concatenate([offset, [0]]))

        angle = rng.random(1)[0] * (MAX_ANGLE - MIN_ANGLE) + MIN_ANGLE
        distance = rng.random(1)[0] * ( MAX_DISTANCE - MIN_DISTANCE) + MIN_DISTANCE

        komo = look_with_angle(C, 'target', distance, angle)
        ret = ry.NLP_Solver(komo.nlp(), verbose=-1).solve()
        if not ret.feasible:
            continue
        path = komo.getPath()

        bot.moveTo(path[-1])
        bot.wait(C)
        bot.hold(floating=False)

        coords = dict()
        count = dict()
        joint_states = []
        for _ in range(IMAGES_PER_POSE):
            joint_states.append(C.getJointState())
            rgb, depth = bot.getImageAndDepth("l_cameraWrist")
            corners, ids, _ = aruco.detectMarkers(rgb, aruco_dict)
            if ids is None:
                continue
            for id, corner in zip(ids.flatten(), corners):
                pixel_coord = corner[0, 0, :].astype(int)
                d = bilinear_depth_interpolation(depth, pixel_coord[0], pixel_coord[1])
                if id not in coords:
                    coords[id] = np.concatenate([corner[0, 0, :], [d]])
                    count[id] = 1
                else:
                    coords[id] += np.concatenate([corner[0, 0, :], [d]])
                    count[id] += 1
            
        if len(coords) == 0:
            continue
        
        corners = []
        ids = []
        for id in coords:
            if count[id] < 3:
                continue
            coords[id] /= count[id]
            corners.append(coords[id])
            ids.append(id)
            marker_set.add(id)

        print(f'dataset {i}, {ids=}')
        h5.write(f'dataset_{i}/joint_state', np.mean(joint_states, axis=0), dtype='float64')
        h5.write(f'dataset_{i}/marker_positions', np.array(corners), dtype='float32')
        h5.write(f'dataset_{i}/marker_ids', np.array(ids), dtype='int32')

        i += 1
        
    del bot
    del C

    manifest = {
    'description': 'for various poses: joint state of the panda and ids and and positions (first corner) of arUco markers as (p_x, p_y, d) coordinates. The parameters entry contains the parameters used for data collection.',
    'n_datasets': NUMBER_OF_POSES,
    'marker_ids': list(marker_set),
    'keys': ['manifest', 'dataset_[i]/joint_state', 'dataset_[i]/marker_positions', 'dataset_[i]/marker_ids'],
    'parameters': {
        'NUMBER_OF_POSES': NUMBER_OF_POSES,
        'IMAGES_PER_POSE': IMAGES_PER_POSE,
        'MIN_DISTANCE': MIN_DISTANCE,
        'MAX_DISTANCE': MAX_DISTANCE,
        'MIN_ANGLE': MIN_ANGLE,
        'MAX_ANGLE': MAX_ANGLE,
        'MAX_TARGET_OFFSET': MAX_TARGET_OFFSET,
        'SEED': SEED
        }
    }
    h5.write('manifest', bytearray(json.dumps(manifest), 'utf-8'), dtype='int8')

if __name__ == '__main__':
    main()