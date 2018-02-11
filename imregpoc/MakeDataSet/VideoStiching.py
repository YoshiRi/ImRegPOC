#!/usr/bin/python
# -*- coding: utf-8 -*-
''' Phase Correlation based image matching and registration libraries
'''
__author__ = "Yoshi Ri"
__copyright__ = "Copyright 2017, The University of Tokyo"
__credits__ = ["Yoshi Ri"]
__license__ = "BSD"
__version__ = "1.0.1"
__maintainer__ = "Yoshi Ri"
__email__ = "yoshiyoshidetteiu@gmail.com"
__status__ = "Production"

# Read Function from upper folder
import sys
sys.path.append('../')
# Each Function are at different Part
import imregpoc
import cv2

class VideoStiching():

    def __init__(self,videoname):
        vidcap = cv2.VideoCapture(videoname)

        success,image = vidcap.read()
        if not(success):
            print('Cannot open the video!')
            exit(-1)
        self.frames = []
        self.cframes = []
        self.frames.append(self.gray(image))
        self.cframes.append(image)
        self.framenum = 1
        while(vidcap.isOpened()):
            success,image = vidcap.read()
            if success:
                self.framenum += 1
                self.frames.append(self.gray(image))
                self.cframes.append(image)
            else:
                break

    def extract_relationship(self):
        self.xMat = np.zeros((self.framenum,self.framenum),dtype=float)
        self.yMat = np.zeros((self.framenum,self.framenum),dtype=float)
        self.thMat = np.zeros((self.framenum,self.framenum),dtype=float)
        self.sMat = np.zeros((self.framenum,self.framenum),dtype=float)
        self.PeakMat = np.zeros((self.framenum,self.framenum),dtype=float)

        for i in range (0,self.framenum-1):
            match = imregpoc.imregpoc(self.frames[i],self.frames[i+1])
            x,y,th,s = match.getParam()
            peak = match.getPeak()
            self.xMat[i,i+1] = x
            self.yMat[i,i+1] = y
            self.thMat[i,i+1] = th
            self.sMat[i,i+1] = s
            self.PeakMat[i,i+1] = peak

            for j in range (i+1,self.framenum):
                match.match_new(self.frames[j])
                x,y,th,s = match.getParam()
                peak = match.getPeak()
                self.xMat[i,j] = x
                self.yMat[i,j] = y
                self.thMat[i,j] = th
                self.sMat[i,j] = s
                self.PeakMat[i,j] = peak

    def gray(self,frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return gray

    def save_data(self):


    def load_data(self):
