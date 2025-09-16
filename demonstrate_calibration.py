import robotic as ry
import numpy as np
from matplotlib.pyplot import get_cmap

camera = 'l_cameraWrist'

C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))
bot = ry.BotOp(C, True)
pcl = C.addFrame("pcl")
bot.getImageAndDepth('l_cameraWrist') # initialize camera
print('Camera Initialized')
C.addFrame('cam_coords', camera).setShape(ry.ST.marker, [.1])

C.getFrame('table').setColor([.2,.2])

plasma = get_cmap("plasma", 256)

bot.hold(floating=True, damping=False)
while bot.getKeyPressed() != ord('q'):
    rgb, _, points = bot.getImageDepthPcl('l_cameraWrist')

    cam_transform = C.getFrame(camera).getTransform()
    points_global = points.reshape(-1, 3) @ cam_transform[:3, :3].T + cam_transform[:3, 3]
    z_normed = np.clip((points_global[:,2]-0.58)/0.04, 0,1)

    color = (plasma(z_normed)[:,:3]*255).astype(np.uint8)

    pcl.setPointCloud(points_global, color)
    bot.sync(C)

bot.hold(floating=False)
