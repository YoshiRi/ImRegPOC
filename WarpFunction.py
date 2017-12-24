import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math

def Warp_4dof(Img,dx,dy,theta,scale):
    rows,cols = Img.shape
    cs = scale * math.cos(theta)
    sn = scale * math.sin(theta)
    
    Rot = np.float32([[cs, sn, 0],[-sn, cs,0],[0,0,1]])
    center_Trans = np.float32([[1,0,cols/2.0],[0,1,rows/2.0],[0,0,1]])
    center_iTrans = np.float32([[1,0,-cols/2.0],[0,1,-rows/2.0],[0,0,1]])
    cRot = np.dot(np.dot(center_Trans,Rot),center_iTrans)
    Trans = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
    Affine = np.dot(cRot,Trans)
    
    outImg = cv2.warpPerspective(Img,Affine,(cols,rows), cv2.INTER_LINEAR)
    return outImg