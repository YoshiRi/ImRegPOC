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
import math

class VideoStiching:

    def __init__(self,videoname):
        vidcap = cv2.VideoCapture(videoname)
        vnames = videoname.replace('/', '.').split('.')
        self.vname = vnames[-2]
        
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
                print('['+str(i)+','+str(j)+']', end='\r')

    def gray(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray
        
    def save_data(self):
        #import os
        #TARGET_DIR = self.vname
        #if not os.path.isdir(TARGET_DIR):
        #    os.makedirs(TARGET_DIR)
        saveMat = np.concatenate([self.xMat,self.yMat,self.thMat,self.sMat,self.PeakMat], axis=0)
        output = self.vname+'.csv'
        np.savetxt(output,saveMat)
            
    def load_data(self,output=None):
        if output==None:
            saveMat = np.loadtxt(self.vname+'.csv',delimiter=' ')
        else:
            saveMat = np.loadtxt(output,delimiter=',')
        Mats = np.split(saveMat,5,axis=0)
        self.xMat = Mats[0]
        self.yMat = Mats[1]
        self.thMat = Mats[2]
        self.sMat = Mats[3]
        self.PeakMat = Mats[4]
    
    def solve_mat(self,vMat,wMat):
        #hei,wid = wMat.shape
        # step 1: AWA
        diagAvec = wMat.sum(axis=0).T + wMat.sum(axis=1)
        diagA = np.diag(diagAvec[1:])
        tri = wMat[1:,1:]
        A = -tri + diagA - tri.T
        # step 2: AWy
        bmat = wMat * vMat
        Bb = bmat.sum(axis=0).T - bmat.sum(axis=1)
        B = Bb[1:]
        # step 3: AWA^-1 AWy
        v = np.dot(np.linalg.inv(A),B)
        return v

    def check_valid_mat(self,wMat):
        diagAvec = wMat.sum(axis=0).T + wMat.sum(axis=1)
        if len(diagAvec) - np.count_nonzero(diagAvec) > 0:
            print('Bad frames exists!')


    def Optimization(self,threshold=None):
        # 1: get a weight matrix
        if threshold ==None:
            threshold = 0.06
        wMat = self.PeakMat
        wMat[wMat <threshold] = 0

        self.check_valid_mat(wMat)

        # 2-1 optimize theta
        vth = self.solve_mat(self.thMat,wMat)
        vth = np.append([0],vth,axis=0)
        # 2-2 optimize kappa
        logsMat = np.log(self.sMat)
        logsMat[logsMat==-np.inf] = 0
        vlogs = self.solve_mat(logsMat,wMat)
        vs = np.exp(vlogs)
        vs = np.append([1],vs,axis=0)

        # 2-3 optimize x and y
        CtaMap = np.tile(vth.reshape(self.framenum,1),(1,self.framenum))
        ScaleMap = np.tile(vs.reshape(self.framenum,1),(1,self.framenum))
        # conversion matrix
        csMap = np.cos(CtaMap)*ScaleMap
        snMap = np.sin(CtaMap)*ScaleMap
        # convert x and y
        tr_xMat =  self.xMat * csMap + self.yMat * snMap
        tr_yMat = -self.xMat * snMap + self.yMat * csMap
        # solve
        vx = np.append([0],self.solve_mat(tr_xMat,wMat),axis=0)
        vy = np.append([0],self.solve_mat(tr_yMat,wMat),axis=0)
        self.results = np.concatenate([vx,vy,vth,vs],axis=0).reshape(4,self.framenum).T
        
    def load_results(self,fname):
        self.results = np.loadtxt(fname,delimiter=',')

    def show_stitched(self):
        self.match = imregpoc.imregpoc(self.frames[0],self.frames[0])
        hei, wid = self.frames[0].shape
        center =[hei/2,wid/2]
        sxmax = wid-1
        sxmin = 0
        symax = hei-1
        symin = 0
        Perspectives = np.zeros((self.framenum,3,3),dtype=float)
        for i in range (0,self.framenum-1):
            perspect = self.match.poc2warp(center,self.results[i])
            xmin,ymin,xmax,ymax = self.match.convertRectangle(perspect)
            sxmax = max(xmax,sxmax)
            sxmin = min(xmin,sxmin)
            symax = max(ymax,symax)
            symin = min(ymin,symin)
            Perspectives[i] = perspect
        swidth,sheight = sxmax-sxmin+1,symax-symin+1
        xtrans,ytrans = 0-sxmin,0-symin
        Trans = np.float32([1,0,xtrans , 0,1,ytrans, 0,0,1]).reshape(3,3)
        self.panorama = np.zeros((sheight,swidth))
        self.panorama[ytrans:ytrans+hei,xtrans:xtrans+wid] = self.match.ref
        for i in range (1,self.framenum-1):
            newTrans = np.dot(Trans,np.linalg.inv(Perspectives[i]))
            warpedimage = cv2.warpPerspective(self.frames[i],newTrans,(swidth,sheight),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
            mask = cv2.warpPerspective(np.ones((hei,wid)),newTrans,(swidth,sheight),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
            mask[mask < 1] = 0 
            Imask = np.ones((sheight,swidth),dtype=float)-mask
            self.panorama = self.panorama*Imask + warpedimage*mask

            cv2.imshow('panorama',self.panorama/255)
            cv2.waitKey(5)
        cv2.imwrite('panoramaimg.png',self.panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def extract_relationship_FP(self):
        self.xMat = np.zeros((self.framenum,self.framenum),dtype=float)
        self.yMat = np.zeros((self.framenum,self.framenum),dtype=float)
        self.thMat = np.zeros((self.framenum,self.framenum),dtype=float)
        self.sMat = np.zeros((self.framenum,self.framenum),dtype=float)
        self.matchedNum = np.zeros((self.framenum,self.framenum),dtype=float)
        self.inliersNum = np.zeros((self.framenum,self.framenum),dtype=float)

        for i in range (0,self.framenum-1):
            match = imregpoc.TempMatcher(self.frames[i],'SIFT')

            for j in range (i+1,self.framenum):
                param,counts,inlier = match.match(self.frames[j])
                x,y,th,s = param
                self.xMat[i,j] = x
                self.yMat[i,j] = y
                self.thMat[i,j] = th/180*math.pi
                self.sMat[i,j] = s
                self.matchedNum[i,j] = counts
                self.inliersNum[i,j] = inlier
                print('['+str(i)+','+str(j)+']', end='\r')

        saveMat = np.concatenate([self.xMat,self.yMat,self.thMat,self.sMat,self.matchedNum,self.inliersNum], axis=0)
        output = self.vname+'_FP'+'.csv'
        np.savetxt(output,saveMat,delimiter=',')

    def load_FP(self):
        output = self.vname+'_FP'+'.csv'
        readMat = np.loadtxt(output,delimiter=',')
        Mats = np.split(readMat,6,axis=0)
        self.xMat = Mats[0]
        self.yMat = Mats[1]
        self.thMat = Mats[2]
        self.sMat = Mats[3]
        self.matchedNum = Mats[4]
        self.inliersNum = Mats[5]

    def getPeak_FP(self,threshold = 0):
        if threshold == 0:
            threshold = 50
        self.PeakMat = np.copy(self.inliersNum)
        self.PeakMat[self.PeakMat<threshold]=0
        self.PeakMat[self.PeakMat>=threshold]=1