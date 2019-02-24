# python 3.6.2
# OpenCv Version: 3.2.0.8

import cv2
import sys
import numpy
import math
from numpy import matrix


R = numpy.array([[0.902701, 0.051530, 0.427171],
                 [0.182987, 0.852568, -0.489535],
                 [-0.389418, 0.520070, 0.760184]],
                numpy.float32)

rvec = cv2.Rodrigues(R)[0]

cameraMatrix = numpy.array([[-1100.000000, 0.000000, 0.000000],
                            [0.000000, -2200.000000, 0.000000],
                            [0.000000, 0.000000, 1.000000]], numpy.float32)

tvec = numpy.array([12, 16, 21], numpy.float32)

objectPoints = numpy.array([[0.1251, 56.3585, 19.3304],
                            [80.8741, 58.5009, 47.9873],
                            [35.0291, 89.5962, 82.2840],
                            [74.6605, 17.4108, 85.8943],
                            [71.0501, 51.3535, 30.3995],
                            [1.4985, 9.1403, 36.4452],
                            [14.7313, 16.5899, 98.8525],
                            [44.5692, 11.9083, 0.4669],
                            [0.8911, 37.7880, 53.1663],
                            [57.1184, 60.1764, 60.7166]], numpy.float32)

print('Initial ObjectPoints')
print(objectPoints)

imagePoints, jac = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, None)
print('Image Points')
print(imagePoints)

###############################################################################
# Part 1: Compute projection matrix (6.3.1)
###############################################################################
# 1. You first create the A matrix which is a 2n by 12 matrix
aMatrix = numpy.zeros((2 * len(objectPoints), 12), numpy.float32)
for i in range(0, len(objectPoints)):
    x = imagePoints[i][0][0]
    y = imagePoints[i][0][1]
    X = objectPoints[i][0]
    Y = objectPoints[i][1]
    Z = objectPoints[i][2]
    # First Row [ X, Y, Z, 1, 0, 0, 0, 0, -xX, -xY, -xZ, -x ]
    rowOne = numpy.array([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x], numpy.float32)
    aMatrix[i*2] = rowOne

    # Second Row [ 0, 0, 0, 0, X, Y, Z, 1, -yX, -yY, -yZ, -y ]
    rowTwo = numpy.array([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y], numpy.float32)
    aMatrix[(i*2)+1] = rowTwo

# 2. A Transpose * A
prod = numpy.dot(aMatrix.T, aMatrix)

# 3. Find the smallest EigenVectors of the dot product
ret, eigenvalues, eigenvectors = cv2.eigen(prod, True)
eigenVector = eigenvectors[len(eigenvectors) - 1]

projMatrix = numpy.array([[eigenVector[0], eigenVector[1], eigenVector[2], eigenVector[3]],
                          [eigenVector[4], eigenVector[5], eigenVector[6], eigenVector[7]],
                          [eigenVector[8], eigenVector[9], eigenVector[10], eigenVector[11]]], numpy.float32)

print("Projection Matrix")
print(projMatrix)
###############################################################################
# Part 2: Decompose projection matrix (6.3.2)
###############################################################################
mHat = projMatrix

# 1. Find the absolute value of the scale vector |y|
normalizer = math.sqrt((mHat[2][0]**2) + (mHat[2][1]**2) + (mHat[2][2]**2))

# 2. Divide each entry by |y| to normalize projection matrix
for i in range(3):
    for j in range(4):
        mHat[i][j] = mHat[i][j] / normalizer

# 3. Parse the Projection matrix into q vectors
q1 = numpy.array([mHat[0][0], mHat[0][1], mHat[0][2]], numpy.float32)
q2 = numpy.array([mHat[1][0], mHat[1][1], mHat[1][2]], numpy.float32)
q3 = numpy.array([mHat[2][0], mHat[2][1], mHat[2][2]], numpy.float32)
q4 = numpy.array([mHat[0][3], mHat[1][3], mHat[2][3]], numpy.float32)

# 4. Find Ox and Oy where Ox= q1*q3 Oy= q2*q3
oX = numpy.dot(q1.T, q3)
oY = numpy.dot(q2.T, q3)

sig = -1

# 5. Find fx and fy where fx= sqrt(q1*q1 = ox**2 ), fy= sqrt(q2*q2 = oy**2 )
fX = math.sqrt(numpy.dot(q1.T, q1) - oX**2)
fY = math.sqrt(numpy.dot(q2.T, q2) - oY**2)

# 6. Compute the rotation Matrix
rotationMatrix = numpy.array([
    [sig*((oX * mHat[2][0] - mHat[0][0]) / fX), sig*((oX * mHat[2][1] - mHat[0][1]) / fX), sig*((oX * mHat[2][2] - mHat[0][2]) / fX)],
    [sig*((oY * mHat[2][0] - mHat[1][0]) / fY), sig*((oY * mHat[2][1] - mHat[1][1]) / fY), sig*((oY * mHat[2][2] - mHat[1][2]) / fY)],
    [sig * mHat[2][0], sig * mHat[2][1], sig * mHat[2][2]]], numpy.float32)

# 7. Calculate the T Vector
tZ = sig*mHat[2][3]
tX = sig*(oX * tZ - mHat[0][3])/fX
tY = sig*(oY * tZ - mHat[1][3])/fY

# 8. Calculate the Camera Matrix
cMatrix = numpy.zeros((3, 3), numpy.float32)
cMatrix[0][0] = sig * fX
cMatrix[1][1] = sig * fY
cMatrix[2][2] = 1


print("\nGiven Rotation Matrix")
print(R)

print("\nCalculated Rotation")
print(rotationMatrix)

print("\nGiven T-Vector")
print(tvec)

print("\nCalculated T-Vector")
print("[ " + str(tX) + " " + str(tY) + " " + str(tZ) + "]")

print("\nGiven Camera Matrix")
print(cameraMatrix)

print("\nCalculated Camera Matrix")
print(cMatrix)

# Projection Matrix
# [[  2.63596755e-02   1.50508073e-03   1.24744661e-02   3.50359261e-01]
#  [  1.06867962e-02   4.97924201e-02  -2.85889115e-02   9.34336543e-01]
#  [  1.03376642e-05  -1.38060086e-05  -2.01803341e-05  -5.57462568e-04]]
#
# Given Rotation Matrix
# [[ 0.90270102  0.05153     0.42717099]
#  [ 0.182987    0.85256797 -0.489535  ]
#  [-0.38941801  0.52007002  0.76018399]]
#
# Calculated Rotation
# [[ 0.90270191  0.05152792  0.42717001]
#  [ 0.18299223  0.85256946 -0.48952946]
#  [-0.38941655  0.52006799  0.76018685]]
#
# Given T-Vector
# [ 12.  16.  21.]
#
# Calculated T-Vector
# [ 11.9986698113 15.9986353853 20.9994392395]
#
# Given Camera Matrix
# [[ -1.10000000e+03   0.00000000e+00   0.00000000e+00]
#  [  0.00000000e+00  -2.20000000e+03   0.00000000e+00]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
#
# Calculated Camera Matrix
# [[ -1.10000000e+03   0.00000000e+00   0.00000000e+00]
#  [  0.00000000e+00  -2.19999316e+03   0.00000000e+00]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]