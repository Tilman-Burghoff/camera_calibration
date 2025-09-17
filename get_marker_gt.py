import robotic as ry
import numpy as np
from robotic.src import h5_helper
import json

MARKER_IDS = [4, 6, 14] # ids of the markers to record

ry.params_add({
    'bot/useOptitrack': True,
    'optitrack/host': "130.149.82.29",
    'optitrack/filter': .5
})

C = ry.Config()

bot = ry.BotOp(C, True)
bot.sync(C)

template = C.getFrame('template')

while bot.getKeyPressed() != ord('q'):
    bot.sync(C, viewMsg=f'Move template to table origin, then press q')
bot.sync(C)

origin = template.getPosition()
C.addFrame('origin').setPosition(origin).setShape(ry.ST.marker, [.05])
C.addFrame('table', 'origin').setRelativePosition([0,0,-.05]).setShape(ry.ST.box, [2.3, 1.24, .11]).setColor([100,100,100,100])
template_marker = C.addFrame('template_marker', 'origin').setShape(ry.ST.marker, [.5])

results = {}

def opti_to_world(z):
    z = z - origin
    z[2] += .6
    return np.array([-z[1], z[0], z[2]])

for id in MARKER_IDS:
    bot.sync(C)
    while bot.getKeyPressed() != ord('q'):
        z = opti_to_world(template.getPosition()) + np.array([0,0,-.6])
        template_marker.setRelativePosition(z)
        bot.sync(C, viewMsg=f'Move template to marker {id}')
    coords = opti_to_world(C.getFrame('template').getPosition())
    results[id] = coords
    print(f'marker {id} position: {np.round(coords, 3)}')


h5 = h5_helper.H5Writer('marker_gt.h5')
for id, pos in results.items():
    h5.write(f'marker_{id}/position', pos, dtype='float32')

manifest = {
    'description': 'ground truth positions of aruco markers in world frame',
    'n_datasets': len(MARKER_IDS),
    'marker_ids': MARKER_IDS,
}
h5.write('manifest', bytearray(json.dumps(manifest), 'utf-8'), dtype='int8')