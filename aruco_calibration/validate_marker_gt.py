# We want to validate the marker positions measured in get_marker_gt.py by moving the gripper to each marker center.
# This however revealed a systemic error of about 0.6 cm in x 7.9cm in  y direction, likely due to the panda position being wrong in the g-file.
# to correct this on has to add (-0.006, 0.079, 0) to the panda position.

import robotic as ry
from robotic.src import h5_helper

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle_camera.g'))
bot = ry.BotOp(C, True)

h5 = h5_helper.H5Reader('marker_gt_new.h5')
manifest = h5.read_dict('manifest')
print(manifest)
h5.print_info()

for id in manifest['marker_ids']:
    key = f'marker_{id}/position'
    print(f'key: {key}')
    pos = h5.read(key)
    marker = C.addFrame(f'marker_{id}', 'l_panda_base')
    marker.setShape(ry.ST.marker, [.05]).setColor([1,0,0,.5])
    marker.setRelativePosition(pos.tolist() + [0])
    print(f'marker {id} position: {pos}')

C.addFrame('baseframemarker', 'l_panda_base').setShape(ry.ST.marker, [1])
C.view(True)

def ik_marker(C, marker_name):
    komo = ry.KOMO(C, 1,1,0,True)

    komo.addControlObjective([], 0, 1e-1)

    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', marker_name], ry.OT.eq, [[1,0,0],[0,1,0]])
    komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.eq, [0,0,1], [0,0,.64])
    komo.addObjective([], ry.FS.scalarProductXZ, ['l_gripper', 'world'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYZ, ['l_gripper', 'world'], ry.OT.eq)
    ret = ry.NLP_Solver(komo.nlp(), verbose=-1).solve()
    print(ret)
    return komo.getPath()[-1]


for id in manifest['marker_ids']:
    for _ in range(3):
        name = f'marker_{id}'
        goal = ik_marker(C, name)
        bot.moveTo(goal)
        bot.wait(C)
    C.view(True)
bot.home(C)