import robotic as ry
import numpy as np
from robotic.src import h5_helper
import json

# TODO align optitrack and real coordinates
# optitrack origin should be at (1.15, .62, .6) in world frame
# also 90 degrees rotated around z axis
# add code to measure aruco pos

MARKER_IDS = [4, 6, 14]

ry.params_add({
    'bot/useOptitrack': True,
    'optitrack/host': "130.149.82.29",
    'optitrack/filter': .5
})

C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))

bot = ry.BotOp(C, True)
bot.sync(C)

print(C.getFrameNames())
optitrack_origin = C.getFrame('origin_optitrack')
C.getFrame('world').setParent(optitrack_origin).setRelativePosition([0,0,0]).setRelativePoseByText('t(-1.15 -.62 -.6)')

marker = C.getFrame('aruco_marker')
marker_pos = C.addFrame('marker_pos').setShape(ry.ST.marker, [.5])
C.addFrame('origin_pos', 'origin_optitrack').setShape(ry.ST.marker, [.5])

world_transform = C.getFrame('world').getTransform()
global_coords = lambda z: np.array([z[1], -z[0], z[2]])

results = {}

for id in MARKER_IDS:
    bot.sync(C)
    while bot.getKeyPressed() != ord('q'):
        coords = global_coords(marker.getPosition())
        marker_pos.setPosition(coords)
        bot.sync(C, viewMsg=f'Move template to marker {id}')
    results[id] = coords

h5 = h5_helper.H5Writer('marker_gt.h5')
for id, pos in results.items():
    h5.write(f'marker_{id}', pos, dtype='float32')

manifest = {
    'description': 'ground truth positions of aruco markers in world frame',
    'n_markers': len(MARKER_IDS),
    'marker_ids': MARKER_IDS
}
h5.write('manifest', bytearray(json.dumps(manifest), 'utf-8'), dtype='int8')
