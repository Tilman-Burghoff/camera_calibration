import robotic as ry
import numpy as np
from komo_paths import look_at_target

C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))

bot = ry.BotOp(C, True)
target = C.addFrame("target").setPosition([-.5, .2, .65])

path, _ = look_at_target(C, "target", height=.3)

bot.moveTo(path[-1])

rgb, _, points = bot.getImageDepthPcl("l_cameraWrist", globalCoordinates=True)


v_coords = np.linspace(0,1, points.shape[0])
u_coords = np.linspace(0,1, points.shape[1])
u, v = np.meshgrid(u_coords, v_coords)
uv = np.stack([u, v, np.ones_like(u)], axis=-1)
X = uv.reshape(-1, 3)
Y = points.reshape(-1, 3)



P = Y.T @ X @np.linalg.inv(X.T @ X)
del bot
del C

print("z std:", np.std(Y[:,2]))
print("Calibration matrix:")
print(np.round(P,3))