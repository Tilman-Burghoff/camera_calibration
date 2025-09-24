import robotic as ry
import numpy as np
import json
from robotic.src import h5_helper
from cv2 import aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)


NUMBER_OF_POSES = 20
IMAGES_PER_POSE = 10
MIN_ANGLE = 0
MAX_ANGLE = 1/3 * 1/2 * np.pi
MIN_DISTANCE = .2
MAX_DISTANCE = .5
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


def main():
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))

    h5 = h5_helper.H5Reader('marker_gt_fixed.h5')
    manifest = h5.read_dict('manifest')

    for id in manifest['marker_ids']:
        key = f'marker_{id}/position'
        pos = h5.read(key)
        marker = C.addFrame(f'marker_{id}', 'l_panda_base')
        marker.setShape(ry.ST.marker, [.05]).setColor([1,0,0,.5])
        marker.setRelativePosition(pos.tolist() + [0])
        print(f'marker {id} position: {pos}')

    h5 = h5_helper.H5Writer('aruco_data_collection.h5')

    bot = ry.BotOp(C, False)
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
        for i in range(IMAGES_PER_POSE):
            joint_states.append(C.getJointState())
            rgb, depth = bot.getImageAndDepth("l_cameraWrist")
            ids, corners, _ = aruco.detectMarkers(rgb, aruco_dict)
            if ids is None:
                continue
            for id, corner in zip(ids.flatten(), corners):
                if id not in coords:
                    coords[id] = np.concatenate([corner[0].flatten(), depth[corner[0]]])
                    count[id] = 1
                else:
                    coords[id] += np.concatenate([corner[0].flatten(), depth[corner[0]]])
                    count[id] += 1
            
        if len(coords) == 0:
            continue
        
        corners = []
        ids = []
        for id in coords:
            coords[id] /= count[id]
            corners.append(coords[id])
            ids.append(id)
            marker_set.add(id)

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