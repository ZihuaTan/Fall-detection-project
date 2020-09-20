import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import string

import itertools
class co:
  def __init__(self, x, y):
    self.x = x
    self.y = y



  def func(self):
    print("FROM check: ",self.x,self.y)

def checkfall():
    namelist = ["0Nose",  "1Neck","2RShoulder",  "3RElbow",  "4RWrist",  "5LShoulder",  "6LElbow",  "7LWrist","8MidHip",  "9RHip", "10RKnee", "11RAnkle", "12LHip", "13LKnee", "14LAnkle", "15REye", "16LEye", "17REar", "18LEar", "19LBigToe", "20LSmallToe", "21LHeel", "22RBigToe", "23RSmallToe", "24RHeel"]
    img = cv2.imread('program/initial/ground.jpg',1)
    img2 = cv2.imread('program/lyi_data/2920.jpg',1)
    result = 'no result'

    jsonpaths = []

    for root, dirs, files in os.walk("program/lying_json", topdown=False):
      for name in files:
        path = os.path.join(root, name)
        if path.endswith("json"): # We want only the images
          jsonpaths.append(path)

    for jsonfile in jsonpaths:
        with open(jsonfile) as f:
          data = json.load(f)



          people = data['people']
          if len(people) == 1:

            for index in people:
             j = index['pose_keypoints_2d']
             index2 = 0
             for i in range(0, len(j), 3):
               p1 = j[i]
               p2 = j[i + 1]

               namelist[index2] = co(p1, p2)

               index2 += 1

            renum = [2,1,5,9,8,12]
            fallindex = 0;
            for i in renum:
                x = round(namelist[i].x)
                y = round(namelist[i].y)
                if x != 0 or y != 0:
                    cv2.circle(img2, (x, y), 10, (0, 0, 0), -1)

                    k = img[y, x]
                    if k[1] == 0 and k[2] ==0:
                        fallindex += 1




            if fallindex >= 3:
                print("FROM check: ","failing")
                result = 'failling'
            else:
                print("FROM check: ","sleep")
                result = 'sleep'


            '''cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image2', 1920, 1080)
            cv2.imshow('image2', img2)'''



          else:
            print("no people or more than 1")
            result = 'no one or more by PE'
    os.remove('program/lying_json/cap_piture_keypoints.json')
    return result



#cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image2', 1920,1080)
#img_2 = cv2.imread('ground.jpg', 1)
#img_1 = cv2.imread('4309.png',1,)

#cv2.imshow('image2', img_1)
'''cv2.waitKey(0)
cv2.destroyAllWindows()'''
