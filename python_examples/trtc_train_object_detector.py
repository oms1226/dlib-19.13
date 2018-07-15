# -*- coding: utf-8 -*-
#!/usr/bin/python

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

def mkdirs(fullpathName):
    dir = os.path.dirname(fullpathName)
    # create directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)

def getFolderName4TraingOptions():
    pass

class trtc_train_object_detector4SVM ():
    mTrainerFullPath = None
    mTargetXmlFullPath = None
    mSvmSaveFullPath = None
    mTraing_options = None

    def __init__(self, trainerFullPath = None, targetXmlFullPath = None, svmSaveFullPath = None, traing_options = None):
        pass

    def getStringTraingOptions(self):
        pass

    def process(self):
        os.system(self.mTrainerFullPath + ' ' + '-t ' + self.mTargetXmlFullPath + ' ' + self.getStringTraingOptions())

        if os.path.isfile('object_detector.svm'):
            mkdirs(self.mSvmSaveFullPath)
            shutil.move(object_detector.svm, self.mSvmSaveFullPath)
        pass

if __name__ == "__main__":
    pass


