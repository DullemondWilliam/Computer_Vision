# python 3.6.2
# OpenCv Version: 3.2.0.8

import cv2
import numpy as np

def order_points(points):
    pts = np.array([[points[0][0][0],points[0][0][1]], [points[1][0][0],points[1][0][1]], [points[2][0][0],points[2][0][1]], [points[3][0][0],points[3][0][1]]], dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

images = [ #cv2.imread('speed_40.bmp', 0),
    # cv2.imread('speed_80.bmp', 0),
    'speedsign3.jpg',
    'speedsign4.jpg',
    'speedsign5.jpg',
    'speedsign12.jpg',
    'speedsign13.jpg',
    'speedsign14.jpg',
    'stop4.jpg'
]
font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(images)):
    img = cv2.imread(images[i], 1)
    cv2.imshow("i", img)
    cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))

    canny = cv2.Canny(blur, 120, 240)

    # cv2.imshow("poly", canny)
    # cv2.waitKey(0)

    # ret, thresh = cv2.threshold(blur, 127, 255, 0)

    # cv2.imshow("poly", canny)
    # cv2.waitKey(0)

    im2, contours, hierarchy = cv2.findContours(canny, 1, cv2.CHAIN_APPROX_SIMPLE)

    t1 = img.copy()
    cv2.drawContours(t1, contours, -1, (0, 255, 0), 1)

    # cv2.imshow("poly", t1)
    # cv2.waitKey(0)

    largest = contours[0];
    large = cv2.contourArea(cv2.convexHull(contours[0]))
    for cnt in contours:
        if cv2.contourArea(cv2.convexHull(cnt)) > cv2.contourArea(cv2.convexHull(largest)):
            largest = cnt
            large = cv2.contourArea(cv2.convexHull(cnt))

    # If there is an area with between .7 and .9 percent of the area of the largest sign then we probably grabbed
    # the whole sign instead of the numbers
    for cnt in contours:
        if cv2.contourArea(cv2.convexHull(cnt)) > large * .7 and cv2.contourArea(cv2.convexHull(cnt)) < large * .9:
            largest = cnt
            break;



    hull = cv2.convexHull(largest)
    epsilon = 0.82 * len(hull)

    approx = cv2.approxPolyDP(largest, epsilon, True)
    t2 = img.copy();
    # cv2.drawContours(t2, approx, -1, (0, 255, 0), 5)

    if len(approx) == 4:

        cv2.drawContours(t2, approx, -1, (0, 255, 0), 5)

        cv2.imshow("poly", t2)
        cv2.waitKey(0)

        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

        dst = np.array([
            [0, 0],
            [200 - 1, 0],
            [200 - 1, 300 - 1],
            [0, 300 - 1]], dtype="float32")

        # four_point_transform(img, approx)
        M = cv2.getPerspectiveTransform(order_points(np.array(approx)), dst)
        warped = cv2.warpPerspective(gray, M, (200, 300))

        speed40 = cv2.imread('speed_40.bmp', 0)
        speed80 = cv2.imread('speed_80.bmp', 0)

        # result = cv2.matchTemplate( warped, gray, cv2.TM_CCOEFF_NORMED)
        # Apply template Matching
        res = cv2.matchTemplate(warped, speed40, cv2.TM_CCOEFF_NORMED)
        min_val40, max_val40, min_loc40, max_loc40 = cv2.minMaxLoc(res)

        res = cv2.matchTemplate(warped, speed80, cv2.TM_CCOEFF_NORMED)
        min_val80, max_val80, min_loc80, max_loc80 = cv2.minMaxLoc(res)

        print( str(max_val40) + "  " + str(max_val80) )

        if max_val40 > max_val80 and max_val40 > .50:
            cv2.putText(img, '40 sign', (10, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif max_val80 > max_val40 and max_val80 > .50:
            cv2.putText(img, '80 sign', (10, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, 'unknown', (10, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, 'stop sign', (10, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


    cv2.imshow("poly", img)
    cv2.waitKey(0)

    cv2.imwrite( images[i]+"_final.jpg", img )
    # cv2.imshow("poly", res)
    # cv2.waitKey(0)

    cv2.destroyAllWindows()





