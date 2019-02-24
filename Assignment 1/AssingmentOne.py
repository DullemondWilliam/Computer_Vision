import cv2
import numpy

worldCoord = numpy.array([[350, 220, 150]], numpy.float)
rotation = numpy.array([0, 0, 0], numpy.float)
trans = numpy.array([-170.0, -105.0, -70.0], numpy.float)
camera = numpy.array([[-500, 0, 320], [0, -500, 240], [0, 0, 1]], numpy.float)

result = cv2.projectPoints(worldCoord, rotation, trans, camera, None)

print( worldCoord[0], ' maps to ', result[0][0][0])
#[ 350.  220.  150.]  maps to  [-805.   -478.75]