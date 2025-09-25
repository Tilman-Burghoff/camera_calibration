# We want to validate the marker positions measured in get_marker_gt.py by moving the gripper to each marker center.
# This however revealed a systemic error of about 0.6 cm in x 7.9cm in  y direction, likely due to the panda position being wrong in the g-file.
# to correct this on has to add (-0.006, 0.079, 0) to the panda position.

import robotic as ry
from robotic.src import h5_helper

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle_camera.g'))
bot = ry.BotOp(C, True)

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
    marker.setRelativePosition(pos)
    print(f'marker {id} position: {pos}')

C.addFrame('baseframemarker', 'l_panda_base').setShape(ry.ST.marker, [1])
C.view(True)

def ik_marker(C, marker_name):
    komo = ry.KOMO(C, 1,1,0,True)

    komo.addControlObjective([], 0, 1e-1)

    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', marker_name], ry.OT.eq, [[1,0,0],[0,1,0]])
    komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.eq, [0,0,1], [0,0,.65])
    komo.addObjective([], ry.FS.scalarProductXZ, ['l_gripper', 'world'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYZ, ['l_gripper', 'world'], ry.OT.eq)
    ret = ry.NLP_Solver(komo.nlp(), verbose=-1).solve()
    print(ret)
    return komo.getPath()[-1]


for id in [0,1,4,6,11,12,14]:
    for _ in range(3):
        name = f'marker_{id}'
        goal = ik_marker(C, name)
        bot.moveTo(goal)
        bot.wait(C)
    C.view(True)
bot.home(C)