import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from robotic.src.h5_helper import H5Reader

cam_names = ['Table Calibrated Camera', 'Aruco Calibrated Camera', 'Old Calibration']

root = Path(__file__).parent.parent
print(root, Path(__file__))

with open(root / 'data/comparison_results_v2.pkl', 'rb') as f:
    results = pickle.load(f)

h5 = H5Reader(root / 'data/marker_gt_fixed.h5')
manifest = h5.read_dict('manifest')
gt_pos = dict()
for id in manifest['marker_ids']:
    position = h5.read(f'marker_{id}/position')
    gt_pos[id] = np.array(position)

# lets plot the results - combined
f, ax = plt.subplots(1,1, figsize=(10,10))
ax.set_title('Calibration Error Comparison')
ax.set_xlabel('Error along x-axis (m)')
ax.set_ylabel('Error along y-axis (m)')
ax.axis('equal')

# add circles for 1,2 cm error
for r in [0.01, 0.02]:
    circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--')
    ax.add_artist(circle)
ax.set_xlim(-0.025, 0.025)
ax.set_ylim(-0.025, 0.025)
ax.grid(True)


for j, (name, res) in enumerate(zip(cam_names, results)):
    all_points = []
    for id, points in res.items():
        all_points.append(np.array(points) - gt_pos[id])
    points = np.vstack(all_points)
    print('average error for', name, np.mean(np.linalg.norm(points[:2,:], axis=-1)))
    ax.scatter(points[:,0], points[:,1], label=name)

ax.legend()
plt.show()

# lets plot the results
f, axs = plt.subplots(1,2, figsize=(10,10))
# add circles for 1,2,3,4 cm error
for r in [0.01, 0.02]:
    circle1 = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--')
    circle2 = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--')
    axs[0].add_artist(circle1)
    axs[1].add_artist(circle2)

axs[0].grid(True)
axs[1].grid(True)

for j, (name, res) in enumerate(zip(cam_names[:2], results[:2])):
    ax = axs[j]
    ax.set_title(name)
    ax.set_xlabel('Error along x-axis (m)')
    ax.set_ylabel('Error along y-axis (m)')
    ax.axis('equal')
    for id, points in res.items():
        points = np.array(points) - gt_pos[id]
        ax.scatter(points[:,0], points[:,1], label=f'Marker {id}')
    

axs[0].set_xlim(-0.02, 0.02)
axs[0].set_ylim(-0.02, 0.02)
axs[1].set_xlim(-0.02, 0.02)
axs[1].set_ylim(-0.02, 0.02)
axs[1].legend()
plt.tight_layout()
plt.show()

f, axs = plt.subplots(1,2, figsize=(20,10), sharex=True, sharey=True)

axs[0].set_title('x-axis errors')
axs[0].set_xlabel('Error along x-axis (m)')
axs[0].set_ylabel('x-position of marker (m)')
axs[1].set_title('y-axis errors')
axs[1].set_xlabel('Error along y-axis (m)')
axs[1].set_ylabel('y-position of marker (m)')
for id, points in results[0].items():
    points = np.array(points) - gt_pos[id]
    axs[0].scatter(points[:,0], np.repeat(gt_pos[id][0], points.shape[0]), label=f'Marker {id}')
    axs[1].scatter(points[:,1], np.repeat(gt_pos[id][1], points.shape[0]), label=f'Marker {id}')

plt.show()



