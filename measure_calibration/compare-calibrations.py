import numpy as np
import robotic as ry
from robotic.src.h5_helper import H5Reader
import matplotlib.pyplot as plt
from cv2 import aruco

def look_at_marker_ik(C, marker_name, distance=0.2):
    komo = ry.KOMO(C, 1, 1, 0, True)
    komo.addControlObjective([], 0), 1e-1
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.positionDiff, [marker_name, 'l_cameraWrist'], ry.OT.eq, np.eye(3), [0,0,distance])
    komo.addObjective([], ry.FS.scalarProductXZ, ['origin', 'l_cameraWrist'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYZ, ['origin', 'l_cameraWrist'], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductZZ, ['origin', 'l_cameraWrist'], ry.OT.ineq)

    ret = ry.NLP_Solver(komo.nlp(), verbose=-1).solve()
    print(ret)
    return komo.getPath()[-1]

def invert_transform(T):
    T_inv = np.eye(4)
    R_inv = T[:3, :3].T
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -R_inv @ T[:3, 3]
    return T_inv

def cam_to_base(C: ry.Config, cam_frame, coords):
    coords = np.array([*coords, 1])
    cam = C.getFrame(cam_frame)
    base = C.getFrame('l_panda_base')
    return invert_transform(base.getTransform()) @ cam.getTransform() @ coords

C = ry.Config()
C.addFile(ry.raiPath('/scenarios/pandaSingle_camera.g'))
C.addFrame('table_calibrated_camera', 'l_panda_link7').setRelativePose(...) # TODO
C.addFrame('aruco_calibrated_camera', 'l_panda_link7').setRelativePose(...) # TODO

h5 = H5Reader('../data/marker_gt_fixed.h5')
manifest = h5.readdict('manifest')

gt_pos = []
for id in manifest['marker_ids']:
    position = h5.read(f'marker_{id}/position')
    C.addFrame(f'marker_{id}', 'l_panda_base').setPosition(position)
    gt_pos.append(position)
gt_pos = np.array(gt_pos)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)

bot = ry.BotOp(C, True)
bot.getImageAndDepth('l_cameraWrist') # initialize camera

results = [dict(), dict()]

for i in range(3):
    for id in manifest['marker_ids']:
        q = look_at_marker_ik(C, f'marker_{id}')
        bot.moveTo(q)

        base_coords = cam_to_base(C, 'l_cameraWrist', cam_pos)
        print(f'Marker {id}, iteration {i}, camera in base frame: {base_coords[:3]}')

        rgb, _, pcl = bot.getImageDepthPcl('l_cameraWrist')
        corners, ids, _ = aruco.detectMarkers(rgb, aruco_dict)
        if ids is None or id not in ids:
            print(f'Marker {id} not detected!')
            continue
        idx = np.where(ids.flatten() == id)
        corners = corners[idx].squeeze().astype("int")
        corner_points = pcl[corners[:,1],corners[:,0]]
        cam_position = np.average(corner_points, axis=0)
        for j, cam in enumerate(['table_calibrated_camera', 'aruco_calibrated_camera']):
            base_coords = cam_to_base(C, cam, cam_position)
            results[j].setdefault(id, []).append(base_coords[:3])
        
# lets plot the results
f, axs = plt.subplots(1,2, figsize=(10,5))
for j, (name, res) in enumerate(zip(['Table Calibrated Camera', 'Aruco Calibrated Camera'], results)):
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
    circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', label=f'{r*100:.0f} cm error')
    for ax in axs:
        ax.add_artist(circle)
        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-0.05, 0.05)
    ax.grid(True)
axs[1].legend()
plt.tight_layout()
plt.show()