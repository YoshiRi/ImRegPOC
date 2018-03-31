# Phase Correlation to Estimate Pose
import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math
from PhaseCorrelation import *
from WarpFunction import *

class TempMatcher:

    def __init__(self,temp,descriptor = 'ORB'):
        
        # switch detector and matcher
        self.detector = self.get_des(descriptor)
        self.bf =  self.get_matcher(descriptor)# self matcher
        
        if self.detector == 0:
            print("Unknown Descriptor! \n")
            sys.exit()
        
        if len(temp.shape) > 2: #if color then convert BGR to GRAY
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        
        self.template = temp
        #self.imsize = np.shape(self.template)
        self.kp1, self.des1 = self.detector.detectAndCompute(self.template,None)        
        self.kpb,self.desb = self.kp1, self.des1
        self.flag = 0 # homography estimated flag
        self.scalebuf = []
        self.scale = 0
        self.H = np.eye(3,dtype=np.float32)
        self.dH1 = np.eye(3,dtype=np.float32)
        self.dH2 = np.eye(3,dtype=np.float32)
        self.matches = []        
        self.inliers = []        
        self.center = np.float32([temp.shape[1],temp.shape[0]]).reshape([1,2])/2

    def get_des(self,name):
        return {
            'ORB': cv2.ORB_create(nfeatures=500,scoreType=cv2.ORB_HARRIS_SCORE),
            'AKAZE': cv2.AKAZE_create(),
            'KAZE' : cv2.KAZE_create(),
            'SIFT' : cv2.xfeatures2d.SIFT_create(),
            'SURF' : cv2.xfeatures2d.SURF_create()
        }.get(name, 0)  
    
    def get_matcher(self,name): # Binary feature or not 
        return {
            'ORB'  : cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'AKAZE': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'KAZE' : cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'SIFT' : cv2.BFMatcher(),
            'SURF' : cv2.BFMatcher()
        }.get(name, 0)  
    
    def match(self,img):
        if len(img.shape) > 2: #if color then convert BGR to GRAY
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             
        kp2,des2 = self.detector.detectAndCompute(img,None)
        print(len(kp2))
        if len(kp2) < 5:
            return
            
        matches = self.bf.knnMatch(self.des1,des2,k=2)
        good = []
        pts1 = []
        pts2 = []
   
        count = 0
        for m,n in matches:      
            if m.distance < 0.5*n.distance:
                good.append([m])
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(self.kp1[m.queryIdx].pt)
                count += 1

        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        self.flag = 0
        self.show = img
        self.matches.append(count)        
        self.inliner = 0

        if count > 4:
            self.H, self.mask = cv2.findHomography(pts1-self.center, pts2-self.center, cv2.RANSAC,3.0)
            self.inliner = np.count_nonzero(self.mask)

        
        param = self.Getpoc()
        return param, count, self.inliner
        
    def Getpoc(self):
        h,w = self.template.shape
        #Affine = MoveCenterOfImage(self.H,[0,0],[w/2,h/2]) 
        Affine = self.H

        if Affine is None:
            return [0,0,0,1]
        
        # Extraction
        A2 = Affine*Affine
        scale = math.sqrt(np.sum(A2[0:2,0:2])/2.0)
        theta = math.atan2(Affine[0,1],Affine[0,0])

        theta = theta*180.0/math.pi

        Trans = np.dot(np.linalg.inv(Affine[0:2,0:2]),Affine[0:2,2:3])
        return [Trans[0],Trans[1],theta,scale]
