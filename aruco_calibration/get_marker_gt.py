import robotic as ry
import numpy as np
from robotic.src import h5_helper
import json

MARKER_IDS =  [0, 1, 4, 6, 11, 12, 14] # ids of the markers to record
ROUNDS = 3

ry.params_add({
    'bot/useOptitrack': True,
    'optitrack/host': "130.149.82.29",
    'optitrack/filter': .5
})

C = ry.Config()

bot = ry.BotOp(C, True)
bot.sync(C)

template = C.getFrame('template')

panda_base_frame = C.addFrame('l_panda_base').setShape(ry.ST.marker, [.05])
origin_frame = C.addFrame('origin')
C.addFrame('table', 'origin').setRelativePosition([0,0,-.05]).setShape(ry.ST.box, [2.3, 1.24, .11]).setColor([100,100,100,100])
template_marker = C.addFrame('template_marker', 'l_panda_base').setShape(ry.ST.marker, [.5])



def opti_to_world(z):
    z = z - panda_base
    return np.array([-z[1], z[0], z[2]]) # pos[1], -pos[0]

results = []

for round in range(ROUNDS):
    panda_base = C.getFrame('panda_base_opti').getPosition()
    panda_base_frame.setPosition(panda_base)
    origin_frame.setPosition(panda_base - np.array([-0.40070843, -0.25559472,  0]))

    results.append({})

    for id in MARKER_IDS:
        bot.sync(C)
        while bot.getKeyPressed() != ord('q'):
            z = opti_to_world(template.getPosition())
            template_marker.setRelativePosition(z)
            bot.sync(C, viewMsg=f'Move template to marker {id}')
        coords = np.zeros(3)
        for _ in range(10):
            bot.sync(C)
            coords += opti_to_world(template.getPosition())
        results[round][id] = coords / 10
        print(f'marker {id} position: {np.round(coords / 10, 3)}')


h5 = h5_helper.H5Writer('marker_gt_new.h5')
for id in MARKER_IDS:
    pos = np.mean([results[r][id] for r in range(ROUNDS)], axis=0)
    h5.write(f'marker_{id}/position', np.array([pos[1], -pos[0]]), dtype='float32')

manifest = {
    'description': 'ground truth positions of the centers of the aruco markers in l_panda_base frame',
    'n_datasets': len(MARKER_IDS),
    'marker_ids': MARKER_IDS,
    'keys': ['manifest', 'marker_[id]/position']
}
h5.write('manifest', bytearray(json.dumps(manifest), 'utf-8'), dtype='int8')