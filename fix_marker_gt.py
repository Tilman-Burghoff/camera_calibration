import robotic as ry
from robotic.src import h5_helper
import numpy as np
import json

h5 = h5_helper.H5Reader('marker_gt.h5')
h5_fixed = h5_helper.H5Writer('marker_gt_fixed.h5')
manifest = h5.read_dict('manifest')
for marker_id in manifest['marker_ids']:
    key = f'marker_{marker_id}/position'
    pos = h5.read(key)
    fixed_pos = np.array([pos[1], -pos[0], 0])
    h5_fixed.write(key, fixed_pos, dtype='float32')

h5_fixed.write('manifest', bytearray(json.dumps(manifest), 'utf-8'), dtype='int8')