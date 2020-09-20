import cv2
import numpy as np

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1920,1080)


def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_1, (x, y), 75, (255, 0, 0), -1)
        cv2.circle(img_2, (x, y), 75, (255, 0, 0), -1)
        cv2.imshow('image',img_1)
        cv2.imwrite('ground.jpg',img_2)

#img_1 = cv2.imread('106.jpg', 1)
img_1 = cv2.imread('demo.jpg', 1)

img_2 = cv2.imread('ground.jpg',1)
cv2.imshow('image',img_1)

cv2.setMouseCallback('image',click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()