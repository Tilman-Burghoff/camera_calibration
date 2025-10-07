# We want to validate the marker positions measured in get_marker_gt.py by moving the gripper to each marker center.
# This however revealed a systemic error of about 0.6 cm in x 7.9cm in  y direction, likely due to the panda position being wrong in the g-file.
# to correct this on has to add (-0.006, 0.079, 0) to the panda position.

import robotic as ry
from robotic.src import h5_helper
from parse_arucos import parse_arucos
from pathlib import Path
import re

USE_JOINTS = True
Z_HEIGHT = 0.68

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
config_with_joints = """Edit cameraWrist: { Q: [-0.0234024, 0.0460216, 0.166593, 0.393053, 0.00494562, -0.010527, -0.919442] }
Edit l_panda_joint1_origin: { pose: [0, 0, 0.333, 1, 0, 0, 4.49499e-08] }
Edit l_panda_joint2_origin: { pose: [0, 0, 0, 0.707106, -0.707106, 0.000673419, 0.000673419] }
Edit l_panda_joint3_origin: { pose: [0, -0.316, 0, 0.707107, 0.707107, -0.000185814, 0.000185814] }
Edit l_panda_joint4_origin: { pose: [0.0825, 0, 0, 0.7071, 0.7071, 0.00304907, -0.00304907] }
Edit l_panda_joint5_origin: { pose: [-0.0825, 0.384, 0, 0.707097, -0.707097, 0.00371641, 0.00371641] }
Edit l_panda_joint6_origin: { pose: [0, 0, 0, 0.707105, 0.707105, 0.00177004, -0.00177004] }
Edit l_panda_joint7_origin: { pose: [0.088, 0, 0, 0.707107, 0.707107, 8.19741e-09, -8.19741e-09] }
aruco_0(table): { Q: [-0.462927, -0.139256, 0.05076], shape: marker, size: [.05] }
aruco_1(table): { Q: [0.359713, -0.153013, 0.0508287], shape: marker, size: [.05] }
aruco_4(table): { Q: [0.126525, 0.353908, 0.0516191], shape: marker, size: [.05] }
aruco_6(table): { Q: [-0.124805, 0.405355, 0.0499016], shape: marker, size: [.05] }
aruco_11(table): { Q: [0.302758, 0.15373, 0.0514353], shape: marker, size: [.05] }
aruco_12(table): { Q: [-0.33829, 0.191526, 0.0497731], shape: marker, size: [.05] }
aruco_14(table): { Q: [0.0428602, 0.179035, 0.0510892], shape: marker, size: [.05] }
"""

config_wo_joints = """Edit cameraWrist: { Q: [-0.0239713, 0.0481723, 0.16886, 0.39354, 0.00971287, -0.00283292, -0.919252] }
aruco_0(table): { Q: [-0.467069, -0.13817, 0.0477699], shape: marker, size: [.05] }
aruco_1(table): { Q: [0.364024, -0.153294, 0.0491862], shape: marker, size: [.05] }
aruco_4(table): { Q: [0.128957, 0.358889, 0.0524716], shape: marker, size: [.05] }
aruco_6(table): { Q: [-0.125091, 0.411095, 0.0514258], shape: marker, size: [.05] }
aruco_11(table): { Q: [0.306857, 0.156501, 0.0506129], shape: marker, size: [.05] }
aruco_12(table): { Q: [-0.340495, 0.195725, 0.0483349], shape: marker, size: [.05] }
aruco_14(table): { Q: [0.044747, 0.182807, 0.0496587], shape: marker, size: [.05] }"""


pattern = r"(Edit )?([A-Za-z\d_]*)(?:\(([a-z]*)\))?: {[A-Za-z ]*: \[([\d., e-]*)\]"
matches = re.findall(pattern, config_with_joints if USE_JOINTS else config_wo_joints)
for match in matches:
    edit, name, parent, values = match
    pose = [float(v) for v in values.split(',')]
    if edit == 'Edit ':
        C.getFrame(name).setRelativePose(pose)
    else:
        C.addFrame(name, parent).setRelativePosition(pose).setShape(ry.ST.marker, [.05])

bot = ry.BotOp(C, True)

def ik_marker(C, marker_name, cost):
    komo = ry.KOMO(C, 1,1,0,True)

    komo.addControlObjective([], 0, cost)

    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', marker_name], ry.OT.eq, target=[0,0,-0.02]) #, [[1,0,0],[0,1,0]])
    # komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.eq, [0,0,1], [0,0,Z_HEIGHT])
    komo.addObjective([], ry.FS.scalarProductXZ, ['l_gripper', 'world'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYZ, ['l_gripper', 'world'], ry.OT.eq)
    ret = ry.NLP_Solver(komo.nlp(), verbose=-1).solve()
    print(ret)
    return komo.getPath()[-1]

def move_to_next_marker(C, marker_name):
    komo = ry.KOMO(C, 2,10,2,True)

    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 2, 1e-1)

    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)


    komo.addObjective([0.2, 2], ry.FS.position, ['l_gripper'], ry.OT.ineq, [0,0,-1], [0,0,Z_HEIGHT+0.005])
    komo.addObjective([2], ry.FS.position, ['l_gripper'], ry.OT.ineq, [0,0,1], [0,0,Z_HEIGHT+0.05])

    # komo.addObjective([0.75, 1.25], ry.FS.position, ['l_gripper'], ry.OT.eq, [0,0,1], [0,0,.7])
    komo.addObjective([2], ry.FS.positionDiff, ['l_gripper', marker_name], ry.OT.eq, [[1,0,0],[0,1,0]])
    komo.addObjective([2], ry.FS.scalarProductXZ, ['l_gripper', 'world'], ry.OT.eq)
    komo.addObjective([2], ry.FS.scalarProductYZ, ['l_gripper', 'world'], ry.OT.eq)
    komo.addObjective([2], ry.FS.position, ['l_gripper'], ry.OT.eq, order=1)
    ret = ry.NLP_Solver(komo.nlp(), verbose=-1).solve()
    print(ret)
    return komo.getPath()


for id in [0,1,4,6,11,12,14]:
    print(f'moving to marker {id}')
    C.view(True)
    name = f'aruco_{id}'
    path = move_to_next_marker(C, name)
    bot.moveAutoTimed(path, 0.5, 0.5)
    bot.wait(C)
    for i in range(5):
        goal = ik_marker(C, name, cost=(i+1))
        bot.moveTo(goal)
        bot.wait(C)

print("done")
C.view(True)
bot.home(C)