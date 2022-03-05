import cmath
import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# initialized the video
cap = cv2.VideoCapture('Videos/test1.mp4')

# colour finder
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 134, 'vmin': 0, 'hmax': 179, 'smax': 234, 'vmax': 255}

# variables for center points
posListX, posListY = [], []
xList = [item for item in range(0, 1920)]
predict = False

while True:
    # Grab the image
    success, img = cap.read()
    # img=cv2.imread("BALL.png")

    # crop the image
    #img = img[0:900, :]

    # identify the color
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # location of ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=300)

    # frame by frame point
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # path with polynomial regression
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, pos, 10, (0, 0, 255), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (255, 0, 0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (255, 0, 0), 2)

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (0, 200, 255), cv2.FILLED)
            #print('y=',y)

        if len(posListX)<10:

            # checking basket
            # X= 1370 - 1620 , Y - 637
            # find x values in y=637 value
            a = A
            b = B
            c = C - 400

            x = int(- b + cmath.sqrt(b ** 2 - (4 * a * c)) / (2 * a))
            predict = 500 < x < 600
            print(x)

        if predict:
        #if 1371 < x < 1587:
            #print("Basket")
            cvzone.putTextRect(imgContours,"Basket",(150,600),scale=7,thickness=5,colorR=(0,200,0),offset=20)
        else:
            #print("Outside")
            cvzone.putTextRect(imgContours, "Outside", (150, 600), scale=7, thickness=5, colorR=(0, 0, 200),offset=20)

    # display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.6, 0.6)
    # cv2.imshow("Image",img)
    cv2.imshow("ImageColor", imgContours)
    cv2.waitKey(90)
