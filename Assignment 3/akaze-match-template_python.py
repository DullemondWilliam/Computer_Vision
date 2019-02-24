# python 3.6.2
# OpenCv Version: 3.2.0.8

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('keble_a_half.bmp',0)          # queryImage
img2 = cv2.imread('keble_b_long.bmp',0) # trainImage

detector = cv2.AKAZE_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)

# cv2.imshow('half', img1)
# cv2.imshow('long', img2)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####################################################
# Creating the warped image
####################################################
hom, status = cv2.findHomography(src_pts, dst_pts)
warped = cv2.warpPerspective(img1, hom, (1082, 440))

# cv2.imshow('warped', warped)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####################################################
# Or the Images
####################################################

imgOr = cv2.bitwise_or(img2, warped)

cv2.imshow('warped', warped)
cv2.imwrite("warped.jpg", warped)


cv2.imshow('OR', imgOr)
cv2.imwrite("or.jpg", imgOr)


cv2.waitKey(0)
cv2.destroyAllWindows()

####################################################
# How to get rid of anomalies
####################################################
'''
To get rid of the pictures anomalies requires 3 steps
1- Calculate the intersection points of the 2 images
2- Cut out the intersection from the non warped image
3- Paste the cut onto teh intersection 
'''