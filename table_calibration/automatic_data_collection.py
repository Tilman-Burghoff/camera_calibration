import robotic as ry
import numpy as np
import json


NUMBER_OF_POSES = 20
MIN_ANGLE = 0
MAX_ANGLE = 4/5 * 1/2 * np.pi
MIN_DISTANCE = .2
MAX_DISTANCE = .5
X_BOUNDS = (-.5, .5)
Y_BOUNDS = (.1, .5)
FILTER_PCL_MIN_DISTANCE = 0.2
FILTER_PCL_MAX_DISTANCE = 0.8
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
    bot = ry.BotOp(C, False)
    bot.getImageAndDepth('l_cameraWrist') # initialize camera

    target = C.addFrame('target').setShape(ry.ST.marker, [.1])

    rng = np.random.default_rng(SEED)

    data = []

    min_pos = np.array([X_BOUNDS[0], Y_BOUNDS[0]])
    max_pos = np.array([X_BOUNDS[1], Y_BOUNDS[1]])

    i = 0

    while i < NUMBER_OF_POSES:
        pos = rng.random(2) * (max_pos - min_pos) + min_pos
        target.setPosition(pos.tolist() + [.6])

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

        _,_, points = bot.getImageDepthPcl("l_cameraWrist")
        points = points.reshape(-1,3)
        mask = (np.linalg.norm(points, axis=-1) > FILTER_PCL_MIN_DISTANCE) & (np.linalg.norm(points, axis=-1) < FILTER_PCL_MAX_DISTANCE)
        data.append({
            'id': i,
            'gripper_pose': C.getFrame("l_gripper").getPose().tolist(), 
            'joint_state':C.getJointState().tolist(),
            'pointcloud': points[mask].tolist()
        })

        i += 1
        

    del bot
    del C

    with open("camera_calibration_data.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()