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

# Phase Correlation to Estimate Pose
import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math

class imregpoc:
    def __init__(self,iref,icmp,*,threshold = 0.06, alpha=0.5, beta=0.8, fitting = 'WeightedCOG'):
        self.orig_ref = iref.astype(np.float32)
        self.orig_cmp = icmp.astype(np.float32)
        self.th = threshold
        self.orig_center = np.array(self.orig_ref.shape)/2.0
        self.alpha = alpha
        self.beta = beta
        self.fitting = fitting

        self.param = [0,0,0,1]
        self.peak = 0
        self.affine = np.float32([1,0,0,0,1,0]).reshape(2,3)
        self.perspective = np.float32([1,0,0,0,1,0,0,0,0]).reshape(3,3)

        # set ref, cmp, center
        self.fft_padding()
        self.match()


    def define_fftsize(self):
        refshape = self.orig_ref.shape
        cmpshape = self.orig_cmp.shape
        if not refshape == cmpshape:
            print("The size of two input images are not equal! Estimation could be inaccurate.")
        maxsize = max(max(refshape),max(cmpshape))
        # we can use faster fft window size with scipy.fftpack.next_fast_len
        return maxsize

    def padding_image(self,img,imsize):
        pad_img = np.pad(img,[(0,imsize[0]-img.shape[0]),(0,imsize[1]-img.shape[1])],'constant')
        return pad_img

    def fft_padding(self):
        maxsize = self.define_fftsize()
        self.ref = self.padding_image(self.orig_ref,[maxsize,maxsize])
        self.cmp = self.padding_image(self.orig_cmp,[maxsize,maxsize])
        self.center = np.array(self.ref.shape)/2.0

    def fix_params(self):
        # If you padded to right and lower, perspective is the same with original image 
        self.param = self.warp2poc(perspective = self.perspective, center = self.orig_center)


    def match(self):
        height,width = self.ref.shape
        self.hanw = cv2.createHanningWindow((width, height),cv2.CV_64F)

        # Windowing and FFT
        G_a = np.fft.fft2(self.ref*self.hanw)
        G_b = np.fft.fft2(self.cmp*self.hanw)

        # 1.1: Frequency Whitening  
        self.LA = np.fft.fftshift(np.log(np.absolute(G_a)+1))
        self.LB = np.fft.fftshift(np.log(np.absolute(G_b)+1))
        # 1.2: Log polar Transformation
        cx = self.center[1]
        cy = self.center[0]
        self.Mag = width/math.log(width)
        self.LPA = cv2.logPolar(self.LA, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        self.LPB = cv2.logPolar(self.LB, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

        # 1.3:filtering
        LPmin = math.floor(self.Mag*math.log(self.alpha*width/2.0/math.pi))
        LPmax = min(width, math.floor(self.Mag*math.log(width*self.beta/2)))
        assert LPmax > LPmin, 'Invalid condition!\n Enlarge lpmax tuning parameter or lpmin_tuning parameter'
        Tile = np.repeat([0.0,1.0,0.0],[LPmin-1,LPmax-LPmin+1,width-LPmax])
        self.Mask = np.tile(Tile,[height,1])
        self.LPA_filt = self.LPA*self.Mask
        self.LPB_filt = self.LPB*self.Mask

        # 1.4: Phase Correlate to Get Rotation and Scaling
        Diff,peak,self.r_rotatescale = self.PhaseCorrelation(self.LPA_filt,self.LPB_filt)
        theta1 = 2*math.pi * Diff[1] / height; # deg
        theta2 = theta1 + math.pi; # deg theta ambiguity
        invscale = math.exp(Diff[0]/self.Mag)
        # 2.1: Correct rotation and scaling
        b1 = self.Warp_4dof(self.cmp,[0,0,theta1,invscale])
        b2 = self.Warp_4dof(self.cmp,[0,0,theta2,invscale])
    
        # 2.2 : Translation estimation
        diff1, peak1, self.r1 = self.PhaseCorrelation(self.ref,b1)     #diff1, peak1 = PhaseCorrelation(a,b1)
        diff2, peak2, self.r2 = self.PhaseCorrelation(self.ref,b2)     #diff2, peak2 = PhaseCorrelation(a,b2)
        # Use cv2.phaseCorrelate(a,b1) because it is much faster

        # 2.3: Compare peaks and choose true rotational error
        if peak1 > peak2:
            Trans = diff1
            peak = peak1
            theta = -theta1
        else:
            Trans = diff2
            peak = peak2
            theta = -theta2

        if theta > math.pi:
            theta -= math.pi*2
        elif theta < -math.pi:
            theta += math.pi*2

        # Results
        self.param = [Trans[0],Trans[1],theta,1/invscale]
        self.peak = peak
        self.perspective = self.poc2warp(self.center,self.param)
        self.affine = self.perspective[0:2,:]
        self.fix_params()

    def match_new(self, newImg):
        self.cmp_orig = newImg
        self.fft_padding()
        height,width = self.cmp.shape
        cy,cx = height/2,width/2
        G_b = np.fft.fft2(self.cmp*self.hanw)
        self.LB = np.fft.fftshift(np.log(np.absolute(G_b)+1))
        self.LPB = cv2.logPolar(self.LB, (cy, cx), self.Mag, flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        self.LPB_filt = self.LPB*self.Mask
        # 1.4: Phase Correlate to Get Rotation and Scaling
        Diff,peak,self.r_rotatescale = self.PhaseCorrelation(self.LPA_filt,self.LPB_filt)
        theta1 = 2*math.pi * Diff[1] / height; # deg
        theta2 = theta1 + math.pi; # deg theta ambiguity
        invscale = math.exp(Diff[0]/self.Mag)
        # 2.1: Correct rotation and scaling
        b1 = self.Warp_4dof(self.cmp,[0,0,theta1,invscale])
        b2 = self.Warp_4dof(self.cmp,[0,0,theta2,invscale])
    
        # 2.2 : Translation estimation
        diff1, peak1, self.r1 = self.PhaseCorrelation(self.ref,b1)     #diff1, peak1 = PhaseCorrelation(a,b1)
        diff2, peak2, self.r2 = self.PhaseCorrelation(self.ref,b2)     #diff2, peak2 = PhaseCorrelation(a,b2)
        # Use cv2.phaseCorrelate(a,b1) because it is much faster

        # 2.3: Compare peaks and choose true rotational error
        if peak1 > peak2:
            Trans = diff1
            peak = peak1
            theta = -theta1
        else:
            Trans = diff2
            peak = peak2
            theta = -theta2

        if theta > math.pi:
            theta -= math.pi*2
        elif theta < -math.pi:
            theta += math.pi*2

        # Results
        self.param = [Trans[0],Trans[1],theta,1/invscale]
        self.peak = peak
        self.perspective = self.poc2warp(self.center,self.param)
        self.affine = self.perspective[0:2,:] 
        self.fix_params()
       
        
    def poc2warp(self,center,param):
        cx,cy = center
        dx,dy,theta,scale = param
        cs = scale * math.cos(theta)
        sn = scale * math.sin(theta)
        
        Rot = np.float32([[cs, sn, 0],[-sn, cs,0],[0,0,1]])
        center_Trans = np.float32([[1,0,cx],[0,1,cy],[0,0,1]])
        center_iTrans = np.float32([[1,0,-cx],[0,1,-cy],[0,0,1]])
        cRot = np.dot(np.dot(center_Trans,Rot),center_iTrans)
        Trans = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
        Affine = np.dot(cRot,Trans)
        return Affine

    def warp2poc(self,center,perspective):
        cx,cy = center
        Center = np.float32([[1,0,cx],[0,1,cy],[0,0,1]])
        iCenter = np.float32([[1,0,-cx],[0,1,-cy],[0,0,1]])

        pocmatrix = np.dot(np.dot(iCenter,perspective),Center)
        dxy = np.dot(np.linalg.inv(pocmatrix[0:2,0:2]),pocmatrix[0:2,2])
        scale = np.sqrt(pocmatrix[0,0]**2+pocmatrix[0,1]**2) 
        theta = np.arctan2(pocmatrix[0,1],pocmatrix[0,0])
        return [dxy[0],dxy[1],theta,scale]

    # Waro Image based on poc parameter
    def Warp_4dof(self,Img,param):
        center = np.array(Img.shape)/2
        rows,cols = Img.shape
        Affine = self.poc2warp(center,param)
        outImg = cv2.warpPerspective(Img, Affine, (cols,rows), cv2.INTER_LINEAR)
        return outImg

    def SubpixFitting(self,mat):
        if self.fitting == 'COG':
            TY,TX = self.CenterOfGravity(mat)
        elif self.fitting == 'WeightedCOG':
            TY,TX = self.WeightedCOG(mat)
        elif self.fitting == 'Parabola':
            TY,TX = self.Parabola(mat)
        else:
            print("Undefined subpixel fitting method! Use weighted center of gravity method instead.")
            TY,TX = self.WeightedCOG(mat)

        return [TY,TX]

    # Get peak point
    def CenterOfGravity(self,mat):
        hei,wid = mat.shape
        if hei != wid: # if mat size is not square, there must be something wrong
            print("Skip subpixel estimation!")
            return [0,0]
        Tile=np.arange(wid,dtype=float)-(wid-1.0)/2.0
        Tx = np.tile(Tile,[hei,1]) # Ty = Tx.T
        Sum = np.sum(mat)
        #print(mat)
        Ax = np.sum(mat*Tx)/Sum
        Ay = np.sum(mat*Tx.T)/Sum
        return [Ay,Ax]

    # Weighted Center Of Gravity
    def WeightedCOG(self,mat):
        if mat.size == 0:
            print("Skip subpixel estimation!")
            Res = [0,0]
        else:
            peak = mat.max()
            newmat = mat*(mat>peak/10) # discard information of lower peak
            Res = self.CenterOfGravity(newmat)
        return Res

    # Parabola subpixel fitting
    def Parabola(self,mat):
        hei,wid = mat.shape
        boxsize = 3
        cy = int((hei-1)/2)
        cx = int((wid-1)/2)
        bs = int((boxsize-1)/2)
        box = mat[cy-bs:cy-bs+boxsize,cx-bs:cx-bs+boxsize]
        # [x^2 y ^2 x y 1]
        Tile = np.arange(boxsize,dtype=float)-bs
        Tx = np.tile(Tile,[boxsize,1])
        Ty = Tx.T
        Ones = np.ones((boxsize*boxsize,1),dtype=float)
        x = Tx.reshape(boxsize*boxsize,1)
        y = Ty.reshape(boxsize*boxsize,1)
        x2 = x*x
        y2 = y*y
        A = np.concatenate((x2,y2,x,y,Ones),1)
        # data = A^+ B
        data = np.dot(np.linalg.pinv(A) , box.reshape(boxsize*boxsize,1))
        # xmax = -c/2a, ymax = -d/2b, peak = e - c^2/4a - d^2/4b
        a,b,c,d,e = data.squeeze()
        Ay = -d /2.0/b
        Ax = -c /2.0/a
        self.peak = e - c*c/4.0/a - d*d/4.0/b
        return [Ay,Ax]


    # Phase Correlation
    def PhaseCorrelation(self, a, b):
        height,width = a.shape
        #dt = a.dtype # data type
        # Windowing

        # FFT
        G_a = np.fft.fft2(a*self.hanw)
        G_b = np.fft.fft2(b*self.hanw)
        conj_b = np.ma.conjugate(G_b)
        R = G_a*conj_b
        R /= np.absolute(R)
        r = np.fft.fftshift(np.fft.ifft2(R).real)
        # Get result and Interpolation
        DY,DX = np.unravel_index(r.argmax(), r.shape)
        # Subpixel Accuracy
        boxsize = 5
        box = r[DY-int((boxsize-1)/2):DY+int((boxsize-1)/2)+1,DX-int((boxsize-1)/2):DX+int((boxsize-1)/2)+1] # x times x box
        # subpix fitting
        self.peak = r[DY,DX]
        TY,TX= self.SubpixFitting(box)
        sDY = TY+DY
        sDX = TX+DX
        # Show the result
        return [math.floor(width/2)-sDX,math.floor(height/2)-sDY],self.peak,r

    def MoveCenter(self,Affine,center,newcenter):
        dx = newcenter[1] - center[1]
        dy = newcenter[0] - center[0]
        center_Trans = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
        center_iTrans = np.float32([[1,0,-dx],[0,1,-dy],[0,0,1]])
        newAffine = center_iTrans.dot( Affine.dot(center_Trans))
        return newAffine
        
    def getParam(self):
        return self.param

    def getPeak(self):
        return self.peak

    def getAffine(self):
        return self.affine

    def getPerspective(self):
        return  self.perspective


    def showRotatePeak(self):
        plt.imshow(self.r_rotatescale,vmin=self.r_rotatescale.min(),vmax=self.r_rotatescale.max(),cmap='gray')
        plt.show()

    def showTranslationPeak(self):
        plt.subplot(211)
        plt.imshow(self.r1,vmin=self.r1.min(),vmax=self.r1.max(),cmap='gray')
        plt.subplot(212)
        plt.imshow(self.r2,vmin=self.r2.min(),vmax=self.r2.max(),cmap='gray')
        plt.show()
        
    def showLPA(self):
        plt.imshow(self.LPA,vmin=self.LPA.min(),vmax=self.LPA.max(),cmap='gray')
        plt.show()

    def showLPB(self):
        plt.imshow(self.LPB,vmin=self.LPB.min(),vmax=self.LPB.max(),cmap='gray')
        plt.show()

    def showMAT(self,MAT):
        plt.figure()
        plt.imshow(MAT,vmin=MAT.min(),vmax=MAT.max(),cmap='gray')
        plt.show()
        
    def saveMat(self,MAT,name):
        cv2.imwrite(name,cv2.normalize(MAT,  MAT, 0, 255, cv2.NORM_MINMAX))

    def isSucceed(self):
        if self.peak > self.th:
            return 1
        return 0

    # function around mosaicing
    def convertRectangle(self,perspective=None):
        if perspective == None:
            perspective = self.perspective
        height,width = self.orig_cmp.shape
        rectangles = np.float32([[0,0, 0,width-1, height-1,0, height-1,width-1]]).reshape(1,4,2)
        converted_rectangle = cv2.perspectiveTransform(rectangles,np.linalg.inv(perspective))
        xmax = math.ceil(converted_rectangle[0,:,0].max())
        xmin = math.floor(converted_rectangle[0,:,0].min())
        ymax = math.ceil(converted_rectangle[0,:,1].max())
        ymin = math.floor(converted_rectangle[0,:,1].min())
        return [xmin,ymin,xmax,ymax]

    def stitching(self,perspective=None):
        if perspective == None:
            perspective = self.perspective
        xmin,ymin,xmax,ymax = self.convertRectangle()
        hei,wid = self.orig_ref.shape
        sxmax = max(xmax,wid-1)
        sxmin = min(xmin,0)
        symax = max(ymax,hei-1)
        symin = min(ymin,0)
        swidth,sheight = sxmax-sxmin+1,symax-symin+1
        xtrans,ytrans = 0-sxmin,0-symin
        Trans = np.float32([1,0,xtrans , 0,1,ytrans, 0,0,1]).reshape(3,3)
        newTrans = np.dot(Trans,np.linalg.inv(perspective))
        warpedimage = cv2.warpPerspective(self.orig_cmp,newTrans,(swidth,sheight),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        warpedimage[ytrans:ytrans+hei,xtrans:xtrans+wid] = self.orig_ref
        plt.figure()
        plt.imshow(warpedimage,vmin=warpedimage.min(),vmax=warpedimage.max(),cmap='gray')
        plt.show()

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
        return {# Knnmatch do not need crossCheck
            'ORB'  : cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'AKAZE': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'KAZE' : cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'SIFT' : cv2.BFMatcher(),
            'SURF' : cv2.BFMatcher()
        }.get(name, 0)  
    
    '''
    Do matching based on the descriptor choosed in the constructor.
    Input 1. Compared Image
    Input 2. showflag for matched image
    '''
    def match(self,img,showflag=0):
        if len(img.shape) > 2: #if color then convert BGR to GRAY
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.cmp = img     
        kp2,des2 = self.detector.detectAndCompute(img,None)
        print('Matched Points Number:'+str(len(kp2)))
        if len(kp2) < 5:
            return [0,0,0,1],0,0
            
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
        self.matches.append(count) #?
        self.inliner = 0

        if count > 4:
            self.H, self.mask = cv2.findHomography(pts1-self.center, pts2-self.center, cv2.RANSAC,3.0)
            self.inliner = np.count_nonzero(self.mask)
        
        if showflag:
            img3 = cv2.drawMatchesKnn(self.template, self.kp1, img, kp2, good, None, flags=2)
            plt.imshow(img3,cmap='gray')
        
        param = self.getpoc()
        return param, count, self.inliner

    def getPerspective(self):
        hei,wid = self.template.shape
        cy,cx = hei/2,wid/2
        Trans = np.float32([1,0,cx , 0,1,cy, 0,0,1]).reshape(3,3)
        iTrans= np.float32([1,0,-cx , 0,1,-cy, 0,0,1]).reshape(3,3)
        return np.dot(Trans,np.dot(self.H,iTrans))

    def getpoc(self):
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

    def convertRectangle(self,perspective=None):
        if perspective == None:
            perspective = self.H
        height,width = self.cmp.shape
        rectangles = np.float32([[0,0, 0,width-1, height-1,0, height-1,width-1]]).reshape(1,4,2)
        converted_rectangle = cv2.perspectiveTransform(rectangles,np.linalg.inv(perspective))
        xmax = math.ceil(converted_rectangle[0,:,0].max())
        xmin = math.floor(converted_rectangle[0,:,0].min())
        ymax = math.ceil(converted_rectangle[0,:,1].max())
        ymin = math.floor(converted_rectangle[0,:,1].min())
        return [xmin,ymin,xmax,ymax]

    def stitching(self,perspective=None):
        if perspective == None:
            perspective = self.getPerspective()
        xmin,ymin,xmax,ymax = self.convertRectangle(self.getPerspective())
        hei,wid = self.template.shape
        sxmax = max(xmax,wid-1)
        sxmin = min(xmin,0)
        symax = max(ymax,hei-1)
        symin = min(ymin,0)
        swidth,sheight = sxmax-sxmin+1,symax-symin+1
        xtrans,ytrans = 0-sxmin,0-symin
        Trans = np.float32([1,0,xtrans , 0,1,ytrans, 0,0,1]).reshape(3,3)
        newTrans = np.dot(Trans,np.linalg.inv(perspective))
        warpedimage = cv2.warpPerspective(self.cmp,newTrans,(swidth,sheight),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        warpedimage[ytrans:ytrans+hei,xtrans:xtrans+wid] = self.template        
        plt.figure()
        plt.imshow(warpedimage,vmin=warpedimage.min(),vmax=warpedimage.max(),cmap='gray')
        plt.show()


if __name__ == "__main__":
    # Read image
    ref = cv2.imread('imgs/ref.png',0)
    cmp = cv2.imread('imgs/cmp.png',0)
    plt.imshow(ref,cmap="gray")

    # reference parameter (you can change this)
    match = imregpoc(ref,cmp)
    print(match.peak,match.param)
    match_para = imregpoc(ref,cmp,fitting='Parabola')
    print(match_para.peak,match_para.param)
    match_cog = imregpoc(ref,cmp,fitting='COG')
    print(match_cog.peak,match_cog.param)

    match.stitching()
    match_para.stitching()
    match_cog.stitching()
    
    center = np.array(ref.shape)/2
    persp = match.poc2warp(center,[-5.40E+01,-2.00E+00,9.72E+01/180*math.pi,6.03E-01])
    match.stitching(persp)
    # Save particular Image
    #match.saveMat(match.LPA,'LPA.png')
    #match.saveMat(match.LPA_filt,'LPA_filt.png')
    #match.saveMat(match.LA,'LA.png')
    