import robotic as ry
import numpy as np

# TODO align optitrack and real coordinates
# optitrack origin should be at (1.15, .62, .6) in world frame
# also 90 degrees rotated around z axis
# add code to measure aruco pos


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
while True:
    bot.sync(C)
    C.view()
    coords = global_coords(marker.getPosition())
    marker_pos.setPosition(coords)
    print(coords)
