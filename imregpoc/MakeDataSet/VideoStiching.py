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

class VideoStiching():

    def __init__(self,videoname):
        vidcap = cv2.VideoCapture(videoname)
        success,image = vidcap.read()
        if not(success):
            print('Cannot open the video!')
            exit(-1)
        self.frames = []
        self.framenum = 0
        while(vidcap.isOpened()):
            success,image = vidcap.read()
            if success:
                self.framenum += 1
                self.append(image)
            else:
                break