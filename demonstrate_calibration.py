import robotic as ry

C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))
bot = ry.BotOp(C, True)
pcl = C.addFrame("pcl", "l_cameraWrist")
bot.getImageAndDepth('l_cameraWrist') # initialize camera


bot.hold(floating=True, damping=False)
while bot.getKeyPressed() != ord('q'):
    rgb, _, points = bot.getImageDepthPcl("l_cameraWrist")
    pcl.setPointCloud(points, rgb)
    bot.sync(C, viewMsg="press q to finish")

bot.hold(floating=False)
bot.home(C)