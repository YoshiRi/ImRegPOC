# Phase Correlation to Estimate Pose
import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math

class imregpoc_nowindow:
    def __init__(self,iref,icmp,*,threshold = 0.06, alpha=0.5, beta=0.8):
        self.ref = iref.astype(np.float32)
        self.cmp = icmp.astype(np.float32)
        self.th = threshold
        self.center = np.array(iref.shape)/2.0
        self.alpha = alpha
        self.beta = beta

        self.param = [0,0,0,1]
        self.peak = 0
        self.affine = np.float32([1,0,0,0,1,0]).reshape(2,3)
        self.perspective = np.float32([1,0,0,0,1,0,0,0,0]).reshape(3,3)

        self.match()

        
    def match(self):
        height,width = self.ref.shape
        self.hanw = np.ones((height,width),dtype='float32')

        # Windowing and FFT
        G_a = np.fft.fft2(self.ref)
        G_b = np.fft.fft2(self.cmp)

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

    def match_new(self, newImg):
        self.cmp = newImg
        height,width = self.cmp.shape
        cy,cx = height/2,width/2
        G_b = np.fft.fft2(self.cmp)
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


    # Waro Image based on poc parameter
    def Warp_4dof(self,Img,param):
        center = np.array(Img.shape)/2
        rows,cols = Img.shape
        Affine = self.poc2warp(center,param)
        outImg = cv2.warpPerspective(Img, Affine, (cols,rows), cv2.INTER_LINEAR)
        return outImg

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
        #TY,TX= CenterOfGravity(box)
        TY,TX= self.WeightedCOG(box)
        sDY = TY+DY
        sDX = TX+DX
        # Show the result
        return [width/2-sDX,height/2-sDY],r[DY,DX],r

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




class imregpoc_noLP:
    def __init__(self,iref,icmp,*,threshold = 0.06, alpha=0.5, beta=100):
        self.ref = iref.astype(np.float32)
        self.cmp = icmp.astype(np.float32)
        self.th = threshold
        self.center = np.array(iref.shape)/2.0
        self.alpha = alpha
        self.beta = beta

        self.param = [0,0,0,1]
        self.peak = 0
        self.affine = np.float32([1,0,0,0,1,0]).reshape(2,3)
        self.perspective = np.float32([1,0,0,0,1,0,0,0,0]).reshape(3,3)

        self.match()

        
    def match(self):
        height,width = self.ref.shape
        self.hanw = cv2.createHanningWindow((height, width),cv2.CV_64F)

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

    def match_new(self, newImg):
        self.cmp = newImg
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


    # Waro Image based on poc parameter
    def Warp_4dof(self,Img,param):
        center = np.array(Img.shape)/2
        rows,cols = Img.shape
        Affine = self.poc2warp(center,param)
        outImg = cv2.warpPerspective(Img, Affine, (cols,rows), cv2.INTER_LINEAR)
        return outImg

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
        #TY,TX= CenterOfGravity(box)
        TY,TX= self.WeightedCOG(box)
        sDY = TY+DY
        sDX = TX+DX
        # Show the result
        return [width/2-sDX,height/2-sDY],r[DY,DX],r

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

class imregpoc_noWCOG:
    def __init__(self,iref,icmp,*,threshold = 0.06, alpha=0.5, beta=0.8):
        self.ref = iref.astype(np.float32)
        self.cmp = icmp.astype(np.float32)
        self.th = threshold
        self.center = np.array(iref.shape)/2.0
        self.alpha = alpha
        self.beta = beta

        self.param = [0,0,0,1]
        self.peak = 0
        self.affine = np.float32([1,0,0,0,1,0]).reshape(2,3)
        self.perspective = np.float32([1,0,0,0,1,0,0,0,0]).reshape(3,3)

        self.match()

        
    def match(self):
        height,width = self.ref.shape
        self.hanw = cv2.createHanningWindow((height, width),cv2.CV_64F)

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

    def match_new(self, newImg):
        self.cmp = newImg
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


    # Waro Image based on poc parameter
    def Warp_4dof(self,Img,param):
        center = np.array(Img.shape)/2
        rows,cols = Img.shape
        Affine = self.poc2warp(center,param)
        outImg = cv2.warpPerspective(Img, Affine, (cols,rows), cv2.INTER_LINEAR)
        return outImg

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
            newmat = mat # discard information of lower peak
            Res = self.CenterOfGravity(newmat)
        return Res

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
        #TY,TX= CenterOfGravity(box)
        TY,TX= self.WeightedCOG(box)
        sDY = TY+DY
        sDX = TX+DX
        # Show the result
        return [width/2-sDX,height/2-sDY],r[DY,DX],r

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


class imregpoc_largeM:
    def __init__(self,iref,icmp,*,threshold = 0.06, alpha=0.5, beta=0.8):
        self.ref = iref.astype(np.float32)
        self.cmp = icmp.astype(np.float32)
        self.th = threshold
        self.center = np.array(iref.shape)/2.0
        self.alpha = alpha
        self.beta = beta

        self.param = [0,0,0,1]
        self.peak = 0
        self.affine = np.float32([1,0,0,0,1,0]).reshape(2,3)
        self.perspective = np.float32([1,0,0,0,1,0,0,0,0]).reshape(3,3)

        self.match()

        
    def match(self):
        height,width = self.ref.shape
        self.hanw = cv2.createHanningWindow((height, width),cv2.CV_64F)

        # Windowing and FFT
        G_a = np.fft.fft2(self.ref*self.hanw)
        G_b = np.fft.fft2(self.cmp*self.hanw)

        # 1.1: Frequency Whitening  
        self.LA = np.fft.fftshift(np.log(np.absolute(G_a)+1))
        self.LB = np.fft.fftshift(np.log(np.absolute(G_b)+1))
        # 1.2: Log polar Transformation
        cx = self.center[1]
        cy = self.center[0]
        self.Mag = width/(math.log(width)-math.log(2)*0.5)
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

    def match_new(self, newImg):
        self.cmp = newImg
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


    # Waro Image based on poc parameter
    def Warp_4dof(self,Img,param):
        center = np.array(Img.shape)/2
        rows,cols = Img.shape
        Affine = self.poc2warp(center,param)
        outImg = cv2.warpPerspective(Img, Affine, (cols,rows), cv2.INTER_LINEAR)
        return outImg

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
            newmat = mat*(mat>peak/10)# discard information of lower peak
            Res = self.CenterOfGravity(newmat)
        return Res

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
        #TY,TX= CenterOfGravity(box)
        TY,TX= self.WeightedCOG(box)
        sDY = TY+DY
        sDX = TX+DX
        # Show the result
        return [width/2-sDX,height/2-sDY],r[DY,DX],r

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


class imregpoc_NoWhite:
    def __init__(self,iref,icmp,*,threshold = 0.06, alpha=0.5, beta=0.8):
        self.ref = iref.astype(np.float32)
        self.cmp = icmp.astype(np.float32)
        self.th = threshold
        self.center = np.array(iref.shape)/2.0
        self.alpha = alpha
        self.beta = beta

        self.param = [0,0,0,1]
        self.peak = 0
        self.affine = np.float32([1,0,0,0,1,0]).reshape(2,3)
        self.perspective = np.float32([1,0,0,0,1,0,0,0,0]).reshape(3,3)

        self.match()

        
    def match(self):
        height,width = self.ref.shape
        self.hanw = cv2.createHanningWindow((height, width),cv2.CV_64F)

        # Windowing and FFT
        G_a = np.fft.fft2(self.ref*self.hanw)
        G_b = np.fft.fft2(self.cmp*self.hanw)

        # 1.1: Frequency Whitening  
        self.LA = np.fft.fftshift(np.absolute(G_a))
        self.LB = np.fft.fftshift(np.absolute(G_b))
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

    def match_new(self, newImg):
        self.cmp = newImg
        height,width = self.cmp.shape
        cy,cx = height/2,width/2
        G_b = np.fft.fft2(self.cmp*self.hanw)
        self.LB = np.fft.fftshift(np.absolute(G_b))
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


    # Waro Image based on poc parameter
    def Warp_4dof(self,Img,param):
        center = np.array(Img.shape)/2
        rows,cols = Img.shape
        Affine = self.poc2warp(center,param)
        outImg = cv2.warpPerspective(Img, Affine, (cols,rows), cv2.INTER_LINEAR)
        return outImg

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
        #TY,TX= CenterOfGravity(box)
        TY,TX= self.WeightedCOG(box)
        sDY = TY+DY
        sDX = TX+DX
        # Show the result
        return [width/2-sDX,height/2-sDY],r[DY,DX],r

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
