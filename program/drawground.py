import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import string

class co:
  def __init__(self, x, y):
    self.x = x
    self.y = y



  def func(self):
    print("FROM Draw: ",self.x,self.y)

def draw():
    namelist = ["0Nose",  "1Neck","2RShoulder",  "3RElbow",  "4RWrist",  "5LShoulder",  "6LElbow",  "7LWrist","8MidHip",  "9RHip", "10RKnee", "11RAnkle", "12LHip", "13LKnee", "14LAnkle", "15REye", "16LEye", "17REar", "18LEar", "19LBigToe", "20LSmallToe", "21LHeel", "22RBigToe", "23RSmallToe", "24RHeel"]
    result = 'FROM Draw: no result'
    jsonpaths = []

    for root, dirs, files in os.walk("program/standing_json", topdown=False):
      for name in files:
        path = os.path.join(root, name)
        if path.endswith("json"): # We want only the images
          jsonpaths.append(path)


    check = 0
    novalue = 0

    for jsonfile in jsonpaths:
        with open(jsonfile) as f:
          data = json.load(f)


          #img_1 = cv2.imread('program/standing_json/444_rendered.png',1)
          img_1 = cv2.imread('program/initial/ground.jpg', 1)

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

            renum = [11,14,19,22]
            for i in renum:
                x = round(namelist[i].x)
                y = round(namelist[i].y)
                if x != 0 or y != 0:
                    #print("FROM Draw: ",x,y)
                    cv2.circle(img_1, (x, y), 75, (255, 0, 0), -1)



            #cv2.imwrite('standing_json/444_rendered.png',img_1)
            cv2.imwrite('program/initial/ground.jpg', img_1)

            result = 'FROM Draw: ground added'
            check +=1



          else:
            novalue +=1
            result = 'FROM Draw: ","no 1'

    os.remove('program/standing_json/cap_piture_keypoints.json')
    return result



'''cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image2', 1920,1080)
img_2 = cv2.imread('initial/ground.jpg', 1)

img_1 = cv2.imread('standing_json/444_rendered.png',1)
cv2.imshow('image2', img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()'''