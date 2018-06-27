#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in a webcam stream using OpenCV.
#   It is also meant to demonstrate that rgb images from Dlib can be used with opencv by just
#   swapping the Red and Blue channels.
#
#   You can run this program and see the detections from your webcam by executing the
#   following command:
#       ./opencv_face_detection.py
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.  
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys
import dlib
import cv2
import os


import numpy as np


print(cv2.__version__)  # my version is 3.1.0


def extract_image(video_source_path):
    count = 0
    color_green = (0, 255, 0)
    line_width = 3
    detector = dlib.get_frontal_face_detector()

    folder = 'C:\\_workspace\dlib-19.13\\trainingDatas\\test'
    try:
        os.mkdir(folder)
    except OSError:
        pass


    angle90 = 90
    angle180 = 180
    angle270 = 270
    scale = 1.0
    cap = cv2.VideoCapture(video_source_path)

    while (cap.isOpened()):
        ret, frame = cap.read()
        # get image height, width
        rows, cols = frame.shape[:2]
        # calculate the center of the image
        # center = (h / 2, w / 2)
        # center = (w / 4, h / 4)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle90, 0.5)
        frame = cv2.warpAffine(frame, M, (cols, rows))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if True:
            dets = detector(gray)
            print dets
            for det in dets:
                cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
            # frame = cv2.resize(frame, (720, 1280))
        cv2.imshow('my movie', frame)

        # cv2.imwrite(os.path.join(folder, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

extract_image('C:\\_workspace\\dlib-19.13\\trainingDatas\\26.mp4')

exit(0)

