# -*- coding: utf-8 -*-
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
import csv
import json
import socket
import sys
import subprocess
import shlex
import traceback

import datetime
import dlib
import cv2
import os
from sys import platform as _platform
import timeit
import collections

import numpy as np


VideoRotationInfos = {
"mix.avi": 0,
}

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

def printExLR (*strs):
    if DEBUG():
        tot = ""
        for string in strs:
            if type(string) is str:
                tot += string
            else:
                tot += str(string)
        print tot,

def printError (*strs):
    tot = ""
    for string in strs:
        if type(string) is str:
            tot += string
        else:
            tot += str(string)
    print tot

def getExceptionString(info):
    return ''.join(traceback.format_exception(*info)[-2:]).strip().replace('\n', ': ')

class evaluate_face_detection4SVM ():
    version = 1.1

    resultS = dict()
    NUMOFFACEDETECTORS = 9
    DEFAULT_RESOLUTION_WIDTH = float(1280)
    DEFAULT_RESOLUTION_HEIGHT = float(720)

    RESULT_TOTAL_FRAMES = 0

    RESULT_SVM_HITCOUNT = 0
    RESULT_SVM_EACH_HITCOUNT = {}

    RESULT_SVM_TRYCOUNT = 0
    RESULT_SVM_EACH_TRYCOUNT = {}

    RESULT_SVM_DURATION = float(0)
    RESULT_SVM_EACH_DURATION = {}

    RESULT_SVM_RECTSIZE = float(0)
    RESULT_SVM_EACH_RECTSIZE = {}

    RESULT_SVM_RELATION_1DEPTH_MATRIX = [[0] * (NUMOFFACEDETECTORS+1) for i in range(NUMOFFACEDETECTORS+1)]

    def __init__(self, videos_dirname = None, detector_dirname = None, traing_options = None):
        if videos_dirname == None:
            if _platform == "linux" or _platform == "linux2" or _platform == "darwin":
                videos_dirname = "../testDatas/videos"
            elif _platform == "win32" or _platform == "win64":
                videos_dirname = "..\\testDatas\\videos"

        if detector_dirname == None:
            if _platform == "linux" or _platform == "linux2" or _platform == "darwin":
                detector_dirname = "../traningOutput/20180423"
            elif _platform == "win32" or _platform == "win64":
                detector_dirname = "..\\traningOutput\\20180423"

        if traing_options == None:
            # train_object_detector_modify.exe -t fd1_keun.xml --u 3 --l 1 --eps 0.05 --p 0 --target-size 6400 --c 700 --n 9 --cell-size 8 --threshold 0.15 --threads 8
            traing_options = {'-t': "fd1_keun.xml", '--u': 3, '--l': 1, '--eps': 0.05, '--p': 0, '--target-size': 6400, '--c': 700, '--n': 9, '--cell-size': 8, '--threshold': 0.15, '--threads': 8}

        printEx(cv2.__version__)  # my version is 3.1.0
        self.resultS['cv2.__version__'] = cv2.__version__
        self.resultS['_version'] = self.version
        self.resultS['start_time'] = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y%m%d%H%M")
        self.resultS['_hostname'] = _platform + "_" + socket.gethostname()
        self.resultS['videos_dirname'] = self.videos_dirname = videos_dirname
        self.resultS['_detector_dirname'] = self.detector_dirname = detector_dirname
        self.resultS['_traing_options'] = json.dumps(traing_options, ensure_ascii=False)
        self.resultS['targetresolution'] = str(self.DEFAULT_RESOLUTION_WIDTH) + "X" + str(self.DEFAULT_RESOLUTION_HEIGHT)
        pass

    def load_detectors(self, dirname):
        reVal = []
        filenames = os.listdir(dirname)
        order = 1
        for filename in filenames:
            if filename.startswith(str(order)):
                full_filename = os.path.join(dirname, filename)
                detector = dlib.fhog_object_detector(full_filename)
                reVal.append(detector)
                printEx (full_filename)
                self.RESULT_SVM_EACH_HITCOUNT[filename] = 0
                self.RESULT_SVM_EACH_TRYCOUNT[filename] = 0
                self.RESULT_SVM_EACH_DURATION[filename] = float(0)
                self.RESULT_SVM_EACH_RECTSIZE[filename] = float(0)
                order += 1
            else:
                break

        self.RESULT_SVM_EACH_HITCOUNT = collections.OrderedDict(sorted(self.RESULT_SVM_EACH_HITCOUNT.items()))
        self.RESULT_SVM_EACH_TRYCOUNT = collections.OrderedDict(sorted(self.RESULT_SVM_EACH_TRYCOUNT.items()))
        self.RESULT_SVM_EACH_DURATION = collections.OrderedDict(sorted(self.RESULT_SVM_EACH_DURATION.items()))
        self.RESULT_SVM_EACH_RECTSIZE = collections.OrderedDict(sorted(self.RESULT_SVM_EACH_RECTSIZE.items()))
        self.resultS['numofdetectors'] = len(reVal)
        return reVal

    def load_targetVideos(self, dirname):
        reVal = []
        self.resultS['videolists'] = ""
        self.resultS['numofvideoss'] = 0

        for (path, dir, files) in os.walk(dirname):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if True:#ext == 'mp4' or ext == 'avi':
                    #print("%s/%s" % (path, filename))
                    full_filename = os.path.join(path, filename)
                    reVal.append(full_filename)
                    printEx(full_filename)
            self.resultS['videolists'] = self.resultS['videolists'] + "," + ",".join(files)
            self.resultS['numofvideoss'] = self.resultS['numofvideoss'] + len(reVal)

        return reVal

    def load_targetVideos_by20180713(self, dirname):
        reVal = []
        filenames = os.listdir(dirname)
        for filename in filenames:
            if filename.endswith("mp4") or filename.endswith("avi"):
                full_filename = os.path.join(dirname, filename)
                reVal.append(full_filename)
                printEx (full_filename)

        self.resultS['videolists'] = ",".join(filenames)
        self.resultS['numofvideoss'] = len(reVal)
        return reVal

    #https://stackoverflow.com/questions/41200027/how-can-i-find-video-rotation-and-rotate-the-clip-accordingly-using-moviepy
    def get_rotation_old(self, file_path_with_file_name):
        """
        Function to get the rotation of the input video file.
        Adapted from gist.github.com/oldo/dc7ee7f28851922cca09/revisions using the ffprobe comamand by Lord Neckbeard from
        stackoverflow.com/questions/5287603/how-to-extract-orientation-information-from-videos?noredirect=1&lq=1

        Returns a rotation None, 90, 180 or 270
        """
        rotation = -1
        cmd = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1"
        args = shlex.split(cmd)
        args.append(file_path_with_file_name)
        ffprobe_output = subprocess.check_output(args).decode('utf-8')

        if len(ffprobe_output) > 0:  # Output of cmdis None if it should be 0
            ffprobe_output = json.loads(ffprobe_output)
            rotation = ffprobe_output

        else:
            rotation = 0

        return (rotation)

    def get_rotation(self, file_path_with_file_name):
        reVal = -1
        cmd = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1"
        args = shlex.split(cmd)
        args.append(file_path_with_file_name)
        fd_popen = subprocess.Popen(args, stdout=subprocess.PIPE).stdout
        #data = fd_popen.read().strip()
        reVal = None
        while 1:
            line = fd_popen.readline()
            if not line:
                break
            elif "0" in str(line) or "90" in str(line) or "180" in str(line) or "270" in str(line):
                reVal = int(str(line).strip("\r").strip("\n").strip())
                if reVal == 90:
                    reVal = 270
                elif reVal == 270:
                    reVal = 90
                break
        fd_popen.close()

        return reVal

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def detection_eachprocess(self, detectors, targetVideo):
        count = 0
        previousIndex = -1
        color_green = (0, 255, 0)
        line_width = 3

        angle = self.get_rotation(targetVideo)

        if angle == -1:
            return

        scale = 1
        forceAngle = False

        if angle == 0 :
            for key in VideoRotationInfos.keys():
                if key in targetVideo:
                    forceAngle = True
                    angle = VideoRotationInfos[key]

        cap = cv2.VideoCapture(targetVideo)

        while (cap.isOpened()):
            ret, frame = cap.read()

            if not ret:
                break

            # get image height, width
            height, width = frame.shape[:2]

#            if forceAngle == False and width > height:
#                frame = self.rotate_bound(frame, 90)
#            else:
            frame = self.rotate_bound(frame, angle)

            height, width = frame.shape[:2]
            x_ratio = float(0)
            y_ratio = float(0)
            if height > width:
                x_ratio = self.DEFAULT_RESOLUTION_WIDTH / height
                y_ratio = self.DEFAULT_RESOLUTION_HEIGHT / width
            else:
                x_ratio = self.DEFAULT_RESOLUTION_HEIGHT / height
                y_ratio = self.DEFAULT_RESOLUTION_WIDTH / width

            #shrink
            frame = cv2.resize(frame, None, fx=x_ratio, fy=y_ratio, interpolation=cv2.INTER_AREA)

            # get image height, width
            height, width = frame.shape[:2]
            printExLR("%s:%s" % ("height", height))
            printExLR("%s:%s" % ("width", width))


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.RESULT_TOTAL_FRAMES += 1

            flag_detect = False
            duration = float(0)
            index = 0
            for detector in detectors:
                start = timeit.default_timer()
                detectResult = detector(gray)
                stop = timeit.default_timer()
                duration += stop - start
                self.RESULT_SVM_DURATION += stop - start
                self.RESULT_SVM_EACH_DURATION[self.RESULT_SVM_EACH_DURATION.keys()[index]] = self.RESULT_SVM_EACH_DURATION[self.RESULT_SVM_EACH_DURATION.keys()[index]] + stop - start
                self.RESULT_SVM_TRYCOUNT += 1
                self.RESULT_SVM_EACH_TRYCOUNT[self.RESULT_SVM_EACH_TRYCOUNT.keys()[index]] = self.RESULT_SVM_EACH_TRYCOUNT[self.RESULT_SVM_EACH_TRYCOUNT.keys()[index]] + 1
                printExLR(detectResult),
                for rect in detectResult:
                    cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), color_green, line_width)
                    cv2.putText(frame, str(duration), (rect.right(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    cv2.putText(frame, self.RESULT_SVM_EACH_HITCOUNT.keys()[index], (rect.left(), rect.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    self.RESULT_SVM_EACH_HITCOUNT[self.RESULT_SVM_EACH_HITCOUNT.keys()[index]] = self.RESULT_SVM_EACH_HITCOUNT[self.RESULT_SVM_EACH_HITCOUNT.keys()[index]] + 1
                    self.RESULT_SVM_HITCOUNT += 1
                    area = abs(rect.left() - rect.right()) * abs(rect.top() - rect.bottom())
                    self.RESULT_SVM_EACH_RECTSIZE  [self.RESULT_SVM_EACH_RECTSIZE  .keys()[index]] = self.RESULT_SVM_EACH_RECTSIZE  [self.RESULT_SVM_EACH_RECTSIZE  .keys()[index]] + area
                    self.RESULT_SVM_RECTSIZE  += area
                    self.RESULT_SVM_RELATION_1DEPTH_MATRIX[index+1][previousIndex+1] += 1
                    flag_detect = True
                    #frame = cv2.resize(frame, (720, 1280))

                    # cv2.imwrite(os.path.join(folder, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file


                cv2.imshow('target video', frame)

                if detectResult:
                    break

                index += 1

            if flag_detect == False:
                previousIndex = -1
            else:
                previousIndex = index

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detection_fullprocess(self, detectors, targetVideos):
        for targetVideo in targetVideos:
            self.detection_eachprocess(detectors, targetVideo)

    def process(self):
        detectors = self.load_detectors(self.detector_dirname)
        targetVideos = self.load_targetVideos(self.videos_dirname)

        if len(detectors) == self.NUMOFFACEDETECTORS:
            printEx(detectors)
            printEx(targetVideos)
        else:
            ArithmeticError("The number of detectors is " + self.NUMOFFACEDETECTORS + ", but " + len(detectors))

        self.detection_fullprocess(detectors, targetVideos)
        self.resultS['end_time'] = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y%m%d%H%M")

        self.resultS['RESULT_TOTAL_FRAMES'] = self.RESULT_TOTAL_FRAMES

        self.resultS['RESULT_SVM_TRYCOUNT'] = self.RESULT_SVM_TRYCOUNT
        self.resultS['RESULT_SVM_EACH_TRYCOUNT'] = json.dumps(self.RESULT_SVM_EACH_TRYCOUNT, ensure_ascii=False)

        self.resultS['RESULT_SVM_HITCOUNT'] = self.RESULT_SVM_HITCOUNT
        self.resultS['RESULT_SVM_EACH_HITCOUNT'] = json.dumps(self.RESULT_SVM_EACH_HITCOUNT, ensure_ascii=False)

        self.resultS['RESULT_SVM_DURATION'] = self.RESULT_SVM_DURATION
        self.resultS['RESULT_SVM_EACH_DURATION'] = json.dumps(self.RESULT_SVM_EACH_DURATION, ensure_ascii=False)

        self.resultS['RESULT_SVM_RECTSIZE'] = self.RESULT_SVM_RECTSIZE
        self.resultS['RESULT_SVM_EACH_RECTSIZE'] = json.dumps(self.RESULT_SVM_EACH_RECTSIZE, ensure_ascii=False)

        self.resultS['RESULT_SVM_RELATION_1DEPTH_MATRIX'] = json.dumps(self.RESULT_SVM_RELATION_1DEPTH_MATRIX, ensure_ascii=False)

##########################################################################################################################################################################
        if self.RESULT_SVM_TRYCOUNT != 0:
            self.resultS['RESULT_SVM_AVG_DURATION'] = self.RESULT_SVM_DURATION / self.RESULT_SVM_TRYCOUNT
        else:
            self.resultS['RESULT_SVM_AVG_DURATION'] = self.RESULT_SVM_DURATION
        RESULT_SVM_AVG_EACH_DURATION = dict()
        for key in self.RESULT_SVM_EACH_DURATION.keys():
            if self.RESULT_SVM_EACH_TRYCOUNT[key] != 0:
                RESULT_SVM_AVG_EACH_DURATION[key] = self.RESULT_SVM_EACH_DURATION[key] / self.RESULT_SVM_EACH_TRYCOUNT[key]
            else:
                RESULT_SVM_AVG_EACH_DURATION[key] = self.RESULT_SVM_EACH_DURATION[key]
        RESULT_SVM_AVG_EACH_DURATION = collections.OrderedDict(sorted(RESULT_SVM_AVG_EACH_DURATION.items()))
        self.resultS['RESULT_SVM_AVG_EACH_DURATION'] = json.dumps(RESULT_SVM_AVG_EACH_DURATION, ensure_ascii=False)

        if self.RESULT_SVM_HITCOUNT != 0:
            self.resultS['RESULT_SVM_AVG_RECTSIZE'] = self.RESULT_SVM_RECTSIZE / self.RESULT_SVM_HITCOUNT
        else:
            self.resultS['RESULT_SVM_AVG_RECTSIZE'] = self.RESULT_SVM_RECTSIZE
        RESULT_SVM_AVG_EACH_RECTSIZE = dict()
        for key in self.RESULT_SVM_EACH_HITCOUNT.keys():
            if self.RESULT_SVM_EACH_HITCOUNT[key] != 0:
                RESULT_SVM_AVG_EACH_RECTSIZE[key] = self.RESULT_SVM_EACH_RECTSIZE[key] / self.RESULT_SVM_EACH_HITCOUNT[key]
            else:
                RESULT_SVM_AVG_EACH_RECTSIZE[key] = self.RESULT_SVM_EACH_RECTSIZE[key]
        RESULT_SVM_AVG_EACH_RECTSIZE = collections.OrderedDict(sorted(RESULT_SVM_AVG_EACH_RECTSIZE.items()))
        self.resultS['RESULT_SVM_AVG_EACH_RECTSIZE'] = json.dumps(RESULT_SVM_AVG_EACH_RECTSIZE, ensure_ascii=False)

##########################################################################################################################################################################
        self.resultS = collections.OrderedDict(sorted(self.resultS.items()))
    def writeResult(self, testresult_fullname = None):
        if testresult_fullname == None:
            RESULT_FILENAME = "result.log"
            if _platform == "linux" or _platform == "linux2" or _platform == "darwin":
                testresult_fullname = "../testResults/" + RESULT_FILENAME

            elif _platform == "win32" or _platform == "win64":
                testresult_fullname = "..\\testResults\\" + RESULT_FILENAME

            testresult_fullname = testresult_fullname + "_" + str(EFD.version)

        mkdirs(testresult_fullname)
        with open(testresult_fullname + ".json", 'a') as f:
            f.write(json.dumps(EFD.resultS) + "\n")
            f.close()

        fWriteKeys = False
        if not os.path.isfile(testresult_fullname + ".csv"):
            fWriteKeys = True

        with open(testresult_fullname + ".csv",'a') as f:
            w = csv.writer(f)
            if fWriteKeys:
                w.writerow(EFD.resultS.keys())
            w.writerow(EFD.resultS.values())
            f.close()


def mkdirs(fullpathName):
    dir = os.path.dirname(fullpathName)
    # create directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    videos_dirname = None
    detector_dirname = None

    if _platform == "linux" or _platform == "linux2" or _platform == "darwin":
        videos_dirname = "../testDatas/videos"
        detector_dirname = "../traningOutput/20180423"
    elif _platform == "win32" or _platform == "win64":
        videos_dirname = "..\\testDatas\\videos"
        detector_dirname = "..\\traningOutput\\20180423"

    # train_object_detector_modify.exe -t fd1_keun.xml --u 3 --l 1 --eps 0.05 --p 0 --target-size 6400 --c 700 --n 9 --cell-size 8 --threshold 0.15 --threads 8
    traing_options = {'-t': "fd1_keun.xml", '--u': 3, '--l': 1, '--eps': 0.05, '--p': 0, '--target-size': 6400, '--c': 700, '--n': 9, '--cell-size': 8, '--threshold': 0.15, '--threads': 8}
    #EFD = evaluate_face_detection4SVM(videos_dirname, detector_dirname, traing_options)
    EFD = evaluate_face_detection4SVM()
    EFD.process()
    EFD.writeResult()

exit(0)
