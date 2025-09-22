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


# lets plot the results
f, axs = plt.subplots(1,2, figsize=(10,5))
for j, (name, res) in enumerate(zip(cam_names[:2], results[:2])):
    ax = axs[j]
    ax.set_title(name)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    for id, points in res.items():
        points = np.array(points) - gt_pos[id]
        ax.scatter(points[:,0], points[:,1], label=f'Marker {id}')
    
# add circles for 1,2,3,4 cm error
for r in [0.01, 0.02, 0.03, 0.04]:
    circle1 = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--')
    circle2 = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--')
    axs[0].add_artist(circle1)
    axs[1].add_artist(circle2)

axs[0].set_xlim(-0.02, 0.02)
axs[0].set_ylim(-0.02, 0.02)
axs[1].set_xlim(-0.02, 0.02)
axs[1].set_ylim(-0.02, 0.02)
axs[1].legend()
plt.tight_layout()
plt.show()

# lets plot the results - combined
f, ax = plt.subplots(1,1, figsize=(5,5))
ax.set_title('Calibration Error Comparison')
ax.set_xlabel('Error along x-axis (m)')
ax.set_ylabel('Error along y-axis (m)')
ax.axis('equal')
for j, (name, res) in enumerate(zip(cam_names, results)):
    all_points = []
    for id, points in res.items():
        all_points.append(np.array(points) - gt_pos[id])
    points = np.vstack(all_points)
    print('average error for', name, np.mean(np.linalg.norm(points[:2,:], axis=-1)))
    ax.scatter(points[:,0], points[:,1], label=name)

# add circles for 1,2,3,4 cm error
for r in [0.01, 0.02, 0.03, 0.04]:
    circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--')
    ax.add_artist(circle)
ax.set_xlim(-0.035, 0.035)
ax.set_ylim(-0.035, 0.035)

ax.grid(True)
ax.legend()
plt.show()