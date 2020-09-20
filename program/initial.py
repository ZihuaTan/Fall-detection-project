import cv2
import numpy as np
import os

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1920,1080)

img_1 = np.zeros([1080,1920,1],dtype=np.uint8)
img_1.fill(255)
cv2.imwrite('initial/ground.jpg',img_1)
print("FROM initial: ",img_1.shape)

'''img = cv2.imread('4309.JPG',1,)
print(img.shape)'''

#cv2.circle(img, (989,1075), 100, (0,0,255),-1)

#x =826
#y =2387

#k = img[y,x]

#print(k)

'''cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''