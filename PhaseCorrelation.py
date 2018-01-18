import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math

# Get peak point
def CenterOfGravity(mat):
    hei,wid = mat.shape
    Tile=np.arange(wid,dtype=float)-(wid-1.0)/2.0
    Tx = np.tile(Tile,[hei,1]) # Ty = Tx.T
    Sum = np.sum(mat)
    #print(mat)
    Ax = np.sum(mat*Tx)/Sum
    Ay = np.sum(mat*Tx.T)/Sum
    return [Ay,Ax]

# Weighted Center Of Gravity
def WeightedCOG(mat):
    if mat.size == 0:
        print("Skip subpixel estimation!")
        Res = [0,0]
    else:
        peak = mat.max()
        newmat = mat*(mat>peak/10)
        Res = CenterOfGravity(newmat)
    return Res

# Phase Correlation
def PhaseCorrelation(a, b):
    height,width = a.shape
    #dt = a.dtype # data type
    # Windowing
    hann_ = cv2.createHanningWindow((height, width),cv2.CV_64F)
    #hann = hann_.astype(dt) # convert to correspoinding dtype
    rhann = np.sqrt(hann_)
    rhann = hann_

    # FFT
    G_a = np.fft.fft2(a*rhann)
    G_b = np.fft.fft2(b*rhann)
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
    TY,TX= WeightedCOG(box)
    sDY = TY+DY
    sDX = TX+DX
    # Show the result
    # print('DX=',width/2-sDX,'DY=',height/2-sDY)
    # print('CorrelationVal=',r[DY,DX])
    plt.imshow(r,vmin=r.min(), vmax=r.max())
    return [width/2-sDX,height/2-sDY],r[DY,DX]
