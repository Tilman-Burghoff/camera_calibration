import robotic as ry

C = ry.Config()
C.addFile(ry.raiPath("scenarios/pandaSingle_camera.g"))
bot = ry.BotOp(C, True)

bot.home(C)
bot.wait(C)

rgb, _, points = bot.getImageDepthPcl("l_cameraWrist", globalCoordinates=True)

pclGlobal =  C.addFrame("pcl")
pclGlobal.setPointCloud(points, rgb)
C.view(True)

pclLocal =  C.addFrame("pclLocal", "l_cameraWrist")
pclLocal.setPointCloud(points, rgb)
C.view(True)