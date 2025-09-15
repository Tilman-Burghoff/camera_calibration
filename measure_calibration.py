import robotic as ry
import numpy as np


def look_at_target(C, target_name, height):
    komo = ry.KOMO(C, 1, 1, 0, True)
    
    komo.addControlObjective([], 0, 1e-1)

    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1])
    komo.addObjective([], ry.FS.positionDiff, ['l_cameraWrist', target_name], ry.OT.eq, np.eye(3), [0,0,height]) 
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    # point camera downwards
    komo.addObjective([], ry.FS.scalarProductXZ, ['world','l_gripper'], ry.OT.eq, [1])
    komo.addObjective([], ry.FS.scalarProductYZ, ['world','l_gripper'], ry.OT.eq, [1])
    # align gripper coords with global coords, used for calibration
    komo.addObjective([], ry.FS.scalarProductXY, ['world','l_gripper'], ry.OT.eq, [1])
    komo.addObjective([],ry.FS.scalarProductXX, ['world','l_gripper'], ry.OT.ineq, [-1])
    
    ry.NLP_Solver(komo.nlp(), verbose=-1).solve()
    return komo.getPath()


def main():
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))

    bot = ry.BotOp(C, True)
    C.addFrame("target").setPosition([-.5, .2, .65])

    path = look_at_target(C, "target", height=.3)

    bot.moveTo(path[-1])

    rgb, _, points = bot.getImageDepthPcl("l_cameraWrist")

    v_coords = np.linspace(0,1, points.shape[0])
    u_coords = np.linspace(0,1, points.shape[1])
    u, v = np.meshgrid(u_coords, v_coords)
    uv = np.stack([u, v, np.ones_like(u)], axis=-1)
    X = uv.reshape(-1, 3)
    cam_transform = C.getFrame('l_cameraWrist').getTransform()
    Y = points.reshape(-1, 3) @ cam_transform[:3, :3].T + cam_transform[:3, 3]
    P = Y.T @ X @ np.linalg.inv(X.T @ X)

    pcl = C.addFrame("pcl", "l_cameraWrist")
    pcl.setPointCloud(points, rgb)
    C.view(True)
    del bot
    del C

    print("z std:", np.std(Y[:,2]))
    print("Calibration matrix:")
    print(np.round(P,3))


if __name__=="__main__":
    main()