import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math

def poc2warp(cx,cy,dx,dy,theta,scale):
    cs = scale * math.cos(theta)
    sn = scale * math.sin(theta)
    
    Rot = np.float32([[cs, sn, 0],[-sn, cs,0],[0,0,1]])
    center_Trans = np.float32([[1,0,cx],[0,1,cy],[0,0,1]])
    center_iTrans = np.float32([[1,0,-cx],[0,1,-cy],[0,0,1]])
    cRot = np.dot(np.dot(center_Trans,Rot),center_iTrans)
    Trans = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
    Affine = np.dot(cRot,Trans)
    return Affine


def Warp_4dof(Img,dx,dy,theta,scale):
    rows,cols = Img.shape
    cy, cx = rows/2.0 ,cols/2.0
    Affine = poc2warp(cy,cx,dx,dy,theta,scale)
    outImg = cv2.warpPerspective(Img, Affine, (cols,rows), cv2.INTER_LINEAR)
    return outImg


def affine2poc(Affine):
    A2 = Affine*Affine
    scale = math.sqrt(np.sum(A2[0:2,0:2])/2.0)
    theta = math.atan2(Affine[1],Affine[0])

    Trans = np.dot(np.linalg.inv(Affine[0:2,0:2]),Affine[1:3,2:3])
    return [Trans[0],Trans[1],theta,scale]

def MoveCenterOfImage(Affine,now,moved):
    dx = moved[0] - now[0]
    dy = moved[1] - now[1]
    center_Trans = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
    center_iTrans = np.float32([[1,0,-dx],[0,1,-dy],[0,0,1]])
    newAffine = center_iTrans.dot( Affine.dot(center_Trans))
    return newAffine