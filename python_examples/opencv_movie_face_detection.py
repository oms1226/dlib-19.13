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
from sys import platform as _platform
import timeit
import collections

import numpy as np

NUMOFFACEDETECTORS = 9
SVM_HIT_COUNT={}

class setting:
    DEBUG = True

def DEBUG():
    return  setting.DEBUG

def printEx (*strs):
    if DEBUG():
        tot = ""
        for string in strs:
            if type(string) is str:
                tot += string
            else:
                tot += str(string)
        print tot

def load_detectors(dirname):
    global SVM_HIT_COUNT
    reVal = []
    filenames = os.listdir(dirname)
    order = 1
    for filename in filenames:
        if filename.startswith(str(order)):
            full_filename = os.path.join(dirname, filename)
            detector = dlib.fhog_object_detector(full_filename)
            reVal.append(detector)
            printEx (full_filename)
            SVM_HIT_COUNT[filename] = 0
            order += 1
        else:
            break
    SVM_HIT_COUNT = collections.OrderedDict(sorted(SVM_HIT_COUNT.items()))

    return reVal

def load_targetVideos(dirname):
    reVal = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        if filename.endswith("mp4") or filename.endswith("avi"):
            full_filename = os.path.join(dirname, filename)
            reVal.append(full_filename)
            printEx (full_filename)


    return reVal



def detection_eachprocess(detectors, targetVideo):
    global SVM_HIT_COUNT
    count = 0
    color_green = (0, 255, 0)
    line_width = 3

    angle90 = 90
    angle180 = 180
    angle270 = 270
    scale = 1.0
    cap = cv2.VideoCapture(targetVideo)

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
        flag_detect = False
        duration = 0.0
        index = 0
        for detector in detectors:
            start = timeit.default_timer()
            detectResult = detector(gray)
            stop = timeit.default_timer()
            printEx(detectResult)
            for rect in detectResult:
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), color_green, line_width)
                cv2.putText(frame, str(stop - start), (rect.right(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(frame, SVM_HIT_COUNT.keys()[index], (rect.left(), rect.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                SVM_HIT_COUNT[SVM_HIT_COUNT.keys()[index]] = SVM_HIT_COUNT[SVM_HIT_COUNT.keys()[index]]+1

                flag_detect = True
                #frame = cv2.resize(frame, (720, 1280))

                # cv2.imwrite(os.path.join(folder, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
                count += 1

            cv2.imshow('target video', frame)

            if detectResult:
                break

            index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(SVM_HIT_COUNT)

    cap.release()
    cv2.destroyAllWindows()

def detection_fullprocess(detectors, targetVideos):
    for targetVideo in targetVideos:
        detection_eachprocess(detectors, targetVideo)



if __name__ == "__main__":
    printEx(cv2.__version__)  # my version is 3.1.0
    detectors = []
    targetVideos = []
    if _platform == "linux" or _platform == "linux2" or _platform == "darwin":
        detectors = load_detectors("../traningOutput/20180423")
        targetVideos = load_targetVideos("../testDatas/videos")
    elif _platform == "win32" or _platform == "win64":
        detectors = load_detectors("..\\traningOutput\\20180423")
        targetVideos = load_targetVideos("..\\testDatas\\videos")
        pass

    if len(detectors) == NUMOFFACEDETECTORS:
        printEx(detectors)
        printEx(targetVideos)

    else:
        ArithmeticError("The number of detectors is " + NUMOFFACEDETECTORS + ", but " + len(detectors))

    detection_fullprocess(detectors, targetVideos)

exit(0)

# detector = dlib.simple_object_detector("C:\\_workspace\\dlib\\resource\\traningOutput\\current\\object_detector_20180418_0_15_front.svm")
#detector = dlib.simple_object_detector("C:\\_workspace\\dlib\\resource\\testbed\\newface\\new_face_keun2.svm")
detector = dlib.fhog_object_detector("C:\\_workspace\\dlib\\resource\\testbed\\newface\\new_face_keun2.svm")

# dlib.train_simple_object_detector("C:\\_workspace\\dlib\\resource\\testbed\\newface\\fd1_keun.xml", "detector.svm", options)
#detector = dlib.simple_object_detector("detector.svm")


def extract_image(video_source_path):
    count = 0
    color_green = (0, 255, 0)
    line_width = 3
    # detector = dlib.get_frontal_face_detector()

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
            printEx(dets)
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
