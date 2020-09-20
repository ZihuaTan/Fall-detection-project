######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

from program import drawground
from program import check
import time
from shutil import copy2

alarm = 0
FPS = 0

if __name__ == "__main__":
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    font = cv2.FONT_HERSHEY_SIMPLEX


    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # Path to image


    # Number of classes the object detector can identify
    NUM_CLASSES = 3

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # od_graph_def = tf.GraphDef()
        od_graph_def = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            # with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()

            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    cap = cv2.VideoCapture(1);

    ret = cap.set(3,1920)
    ret = cap.set(4,1080)
    print("FROM a_pic: ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("FROM a_pic: ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    while(True):
        start_time = time.time()

        ret,frame = cap.read()

        #cv2.imwrite('program/initial/47.jpg', frame)

        cv2.imwrite('program/testimages_forML/cap_piture.jpg', frame)
        image = cv2.imread('program/testimages_forML/cap_piture.jpg')

        showscreen = cv2.resize(frame,(640,360))
        showoutput = cv2.imread('program/initial/white.jpg',1)
        cv2.imshow('picture', showscreen)
        #print("FROM a_pic: write")

        cv2.imshow('output', showoutput)

        #image = cv2.imread('program/backup/cap_piture.jpg')
        #cv2.imwrite('program/testimages_forML/cap_piture.jpg', image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')

        (numdetct, ouput, image) = vis_util.visualize_boxes_and_labels_on_image_array(
            # vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        # All the results have been drawn on image. Now display the image.
        print("FROM a_pic: detect number: ", numdetct, ouput)
        FPS += 1
        MLresult = str(numdetct) + ", " + ouput + str(FPS)

        #numdetct =1;
        #ouput = "lying"
        showoutput = cv2.imread('program/initial/white.jpg', 1)

        showoutput = cv2.putText(showoutput, MLresult, (10, 100), font, 4, (0, 255, 255), 10, cv2.LINE_AA)
        cv2.imshow('output', showoutput)

        if numdetct == 1 or numdetct == 2:
            if ouput == "standing":
                alarm = 0

                print("FROM a_pic: ","stand")
                showoutput = cv2.imread('program/initial/white.jpg', 1)
                showoutput = cv2.putText(showoutput, 'stand', (10, 100), font, 4, (0, 255, 0), 10, cv2.LINE_AA)
                cv2.imshow('output', showoutput)
                #copy2('program/std_data/cap_piture_keypoints.json', 'program/standing_json/cap_piture_keypoints.json')


                os.chdir(r"C:\pose_estimation\openpose")
                os.system(
                   'cmd/c build\\x64\\Release\\OpenPoseDemo.exe --image_dir C:\\tensorflow1\\models\\research\\object_detection\\program\\testimages_forML  --write_images C:\\tensorflow1\\models\\research\\object_detection\\program\\standing_json --display 0 --write_json C:\\tensorflow1\\models\\research\\object_detection\\program\\standing_json')

                os.chdir(CWD_PATH)
                os.remove("program/testimages_forML/cap_piture.jpg")
                ouputresult = drawground.draw()
                print(ouputresult)
                FPS +=1
                ouputresult = str(FPS) + ouputresult
                showoutput = cv2.imread('program/initial/white.jpg', 1)
                showoutput = cv2.putText(showoutput, ouputresult, (10, 100), font, 4, (255, 255, 0), 10, cv2.LINE_AA)
                cv2.imshow('output', showoutput)



            elif ouput == "lying" or ouput == "seat":
                #copy2('program/lyi_data/cap_piture_keypoints.json', 'program/lying_json/cap_piture_keypoints.json')
                print("FROM a_pic: ", "lie")
                showoutput = cv2.imread('program/initial/white.jpg', 1)

                showoutput = cv2.putText(showoutput, 'lie', (10, 100), font, 4, (0, 255, 0), 10, cv2.LINE_AA)
                cv2.imshow('output', showoutput)

                os.chdir(r"C:\pose_estimation\openpose")
                os.system(
                    'cmd/c build\\x64\\Release\\OpenPoseDemo.exe --image_dir C:\\tensorflow1\\models\\research\\object_detection\\program\\testimages_forML  --write_images C:\\tensorflow1\\models\\research\\object_detection\\program\\lying_json --display 0 --write_json C:\\tensorflow1\\models\\research\\object_detection\\program\\lying_json')


                os.chdir(CWD_PATH)

                os.remove("program/testimages_forML/cap_piture.jpg")
                outputlie = check.checkfall()
                FPS +=1
                outputlie = str(FPS) + outputlie
                if outputlie == 'failling':
                    alarm = alarm + 1
                if alarm >= 2:
                    showoutput = cv2.imread('program/initial/white.jpg', 1)
                    showoutput = cv2.putText(showoutput, 'ALARM_NEED_ACTION', (10, 100), font, 4, (0, 0, 255), 10,
                                             cv2.LINE_AA)
                    cv2.imshow('output', showoutput)
                else:

                    print(outputlie)
                    showoutput = cv2.imread('program/initial/white.jpg', 1)
                    showoutput = cv2.putText(showoutput, outputlie, (10, 100), font, 4, (255, 255, 0), 10,
                                             cv2.LINE_AA)
                    cv2.imshow('output', showoutput)



        else:
            print("FROM a_pic: ","no detect or more than one")
            alarm = 0
            showoutput = cv2.imread('program/initial/white.jpg', 1)
            FPS +=1
            nodetect = str(FPS) + 'no detect by ML'
            showoutput = cv2.putText(showoutput,nodetect , (10, 100), font, 4, (0, 255, 0), 10, cv2.LINE_AA)
            cv2.imshow('output', showoutput)
            os.remove("program/testimages_forML/cap_piture.jpg")


        print("FROM a_pic: ","--- %s seconds ---" % (time.time() - start_time))






        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()




'''cv2.imshow('Object detector', image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
