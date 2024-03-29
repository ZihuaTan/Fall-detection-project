# Fall-detection-project
Designed a fall detection project using machine learning.
# Introduction
A fall detection project using Tensorflow Object Detection API, Openpose and OpenCV.
The project will be read an image that is captured from web camera. Then the image will go through a machine learning model to determine whether the person in the image is standing， seating or lying. After that, the image will be processed by Openpose to generate the join keypoints of the human being. These join keypoints will be used to do the fall detection or mark the ground area of the image.
## Project details: step by step
**1. The web camera will captured a picture and store the pictrue to the local assigned foloder. Then the image will be process by a trained machine learning model to detect the posture of the human being in the image.**

<p align="center">
 On the left are the input pictures and on the right are the output pictures of the machine learning model.
  <img src="doc/cap_picture_std.jpg" width="320" height="180">
  <img src="doc/ML_result_for_standing.JPG" width="320" height="180">
</p>
<p align="center">
  <img  src="doc/cap_piture_lying.jpg" width="320" height="180">
  <img  src="doc/ML_result_for_lying.JPG" width="320" height="180">
</p>


**2. The image is sent to the Openpose to generate the join keypoints of human being. This will produce a Json file that contains all the join keypoints of humans in the image.**

<p align="center">
  Pictures show the join keypoints after processing by the Openpose
  <br>
  <img src="doc/cap_picture_std_rendered.png" width="320" height="180">
  <img src="doc/cap_piture_lying_rendered.png" width="320" height="180">
</p>

**3. The Json that is produced by the Openpose contains the xy coordinates of the join keypoints in the image. It will be used for two different purposes: draw ground area and perform fall detection.**

<p align="center">
  Opempose output join keypoints indices
  <br>
  <img src="doc/keypoints_pose_25.png" width="242" height="426">
  <br>
  Source from :https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
</p>

1. Draw ground area:<br>
When the machine learning model detects the person in the image is standing, the Json file will be used to mark the ground area of the image. We will extract the feet coordinates from the Json file and mark these coordinates in the ground.jpg with blue circle. The ground.jpg is a white background image with the same size as the input image.
<p align="center"> 
  <kbd>
  <img src="doc/444_rendered.jpg" width="320" height="180"> 
  <img src="doc/ground.jpg" width="320" height="180">
</kbd>
</p>
The program will keep drawing the ground area when the person is walking around the room. After a while, we will have a completed ground area that was marked in blue in the ground.jpg. 
<p align="center"> 
  <kbd>
  <img src="doc/mark.jpg" width="320" height="180"> 
  <img src="doc/marked_ground.jpg" width="320" height="180">
</kbd>
</p>
2. Fall detection:<br>
After we have the target ground area, we will able to perfrom the fall detection. When the machine learning model detects a person is lying, we will extract the coordinates of shoulders neck and hips from the Json file. Then we will place these coordinates on the ground.jpg file to check whether these coordinates are lying on the target ground area or not. The way we check whether the coordinate is lying on the target region or not is to check the color on that coodinate in ground.jpg. If the color on that coordinate in ground.jpg is blue then the coordinate is lying on the target ground area. When all the 6 coordinates are lying on the target area, it will trigger the fall detection alarm. 
<p align="center"> 
  <kbd>
  <img src="doc/lying_for_git1.jpg" width="320" height="180"> 
  <img src="doc/lying_for_ground1.jpg" width="320" height="180">
   </kbd>
  <br>
  The person is lying outside the target area which will not trigger the fall detection alarm.
</p>
<p align="center"> 
  <kbd>
  <img src="doc/lying_for_git2.jpg" width="320" height="180"> 
  <img src="doc/lying_for_ground2.jpg" width="320" height="180">
   </kbd>
  <br>
  The person is lying in the target area which triggers the fall detection alarm.
</p>

**Notice:<br>**
**1. This project only work for one object whcih means if there are more than one person show up in the camera, the fall detection will not be activate.**
<br>
**2. The camera will need to be fixed to make sure the target ground area will not change.**

## How to build the project:
1.You will need to have a well trained machine learning model to detect the posture of human being. I was using the tutorial from EdjeElectronics to train my Machine learning model for detecting the posture of human being. Link to the tutorial: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10. It is a great tutorial for beginner to start a object detection project. The lastest version of tensorflow is not compatible with this tutorial. I was using tensorflow 1.14 to sucessful run the tutorial. You will need the machine learning model to detect two kind of human postures: lying and standing. So, I prepared around 100 pictures for each postures and label them using labelImg. Link to the labelImg: https://github.com/tzutalin/labelImg. Then I followed the tutorial to finish the training process of my machine learning model.

2.After you got the machine learning model ready, you needed to add the Conda environment into PyCharm so that you could create a project to use the machine learning model. In order to use this machine learning model on PyCharm, you also need to install the Protobufs and compile the protobuf files again. 

3.it is time to build the openpose system. Link to the openpose: https://github.com/CMU-Perceptual-Computing-Lab/openpose. Download the openpose repository to C:\pose_estimation\openpose. Follow the procedures on the Installation section in: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#installation to build the openpose. 

4.Download or clone my repository to C:\tensorflow1\models\research\object_detection

5.Run the a_Picturetake_ML.py file on PyCharm


