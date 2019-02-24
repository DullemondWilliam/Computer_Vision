import cv2
import numpy as np

# python 3.6.2
# OpenCv Version: 3.2.0.8

img = cv2.imread('track.jpg', 1)
blur = cv2.blur(img, (5, 5))

edges = cv2.Canny(blur, 10, 100)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
print( len(lines))
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(255),2)

cv2.imshow("lines",img)

cv2.imwrite("LinesOutput.jpg",img)

cv2.waitKey(0)
