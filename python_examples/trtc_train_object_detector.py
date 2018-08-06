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

def getFolderName4TraingOptions(options):
    reVal = None
    for key in options.keys():
        if reVal != None:
            reVal = reVal + '.'
        if '-t' == key:
            options[key] = '/'.join(str(options[key]).split('/')[:-1])
        newKey = key.replace('-', '_').replace('/', '.').replace('\\', '.')
        value = str(options[key]).replace('-', '_').replace('/', '.').replace('\\', '.')
        if reVal == None:
            reVal = ("%s[%s]" % (newKey, value))
        else:
            reVal = reVal + ("%s[%s]" % (newKey, str(value)))

    return reVal + '_' + (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y%m%d%H%M")

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
    traing_options = {'-t': "../trainingDatas/20180423/1_detector_front.xml", '--u': 3, '--l': 1, '--eps': 0.05, '--p': 0, '--target-size': 6400, '--c': 700, '--n': 9, '--cell-size': 8, '--threshold': 0.15, '--threads': 8}
    print getFolderName4TraingOptions(traing_options)
    pass


