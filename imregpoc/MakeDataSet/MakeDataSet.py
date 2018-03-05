import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math

# Read Function from upper folder
import sys
sys.path.append('../')
# Each Function are at different Part
from WarpFunction import *
from PhaseCorrelation import *
from POC import *

# Center cropping
def cropCenter(Img,Size):
    hei,wid = Img.shape
    cy,cx = int(hei/2) ,int(wid/2) 
    sy,sx = int(Size[0]/2),int(Size[1]/2)
    #print(cy-sy,cy+sy,cx-sx,cx+sx)
    return Img[cy-sy:cy+sy,cx-sx:cx+sx]

def addGaussianNoise(src):
    row,col= src.shape
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = src + gauss

    return noisy

def MakeDataSet(Temp,nS=256,outfolder = 'Data/Test1',DataNum = 100):
    import random
    import os
    if not os.path.exists(outfolder+'/'):
        os.makedirs(outfolder+'/')
    if not os.path.exists(outfolder+'_noised'+'/'):
        os.makedirs(outfolder+'_noised'+'/')

    # Define size : nS = 256
    hei,wid = Temp.shape
    cy,cx = hei/2 ,wid/2 
    
    # Save Reference
    A_ref = poc2warp(cx,cy,0,0,0,1)
    Iref = cv2.warpPerspective(Temp,A_ref,(wid,hei))
    cIref=cropCenter(Iref,[nS,nS])
    cv2.imwrite(outfolder+'/'+'ref.png',cIref)
    cv2.imwrite(outfolder+'_noised'+'/'+'ref.png',cIref)


    # Save Transfered Image
    iterate = 1
    dataset = []
    for iterate in range(1,DataNum+1):
        outname = outfolder +'/'+'cmp' + str(iterate) + '.png'
        outname_n = outfolder +'_noised' +'/' +'cmp' + str(iterate) + '.png'

        rdx = random.randint(-nS/4,nS/4)
        rdy = random.randint(-nS/4,nS/4)
        rCta = random.uniform(-math.pi,math.pi)
        rS = random.uniform(0.5,1.5)
        # Make Image and Save
        A_cmp = poc2warp(cx,cy,rdx,rdy,rCta,rS)
        Icmp = cv2.warpPerspective(Temp,A_cmp,(wid,hei))
        cIcmp=cropCenter(Icmp,[nS,nS])
        cv2.imwrite(outname,cIcmp)
        cv2.imwrite(outname_n,addGaussianNoise(cIcmp))

        # Save template in Txt
        dataset.append([rdx,rdy,rCta*180/math.pi,rS])
    dataset = np.array(dataset,np.float32).reshape(DataNum,4)
    np.savetxt(outfolder+'/'+'TrueParam.csv',dataset,delimiter=',')
    np.savetxt(outfolder+'_noised' +'/' +'TrueParam.csv',dataset,delimiter=',')



def GetFileName():
    # Import module
    import os, tkinter, tkinter.filedialog, tkinter.messagebox
    # Choose File
    root = tkinter.Tk()
    root.withdraw()
    # Any file type of Image
    fTyp = [("",["*.png","*.jpg","*.bmp","*.tif"])]
    iDir = os.path.abspath(os.path.dirname(__file__))
    tkinter.messagebox.showinfo('Image choice','Choose Base Image!')
    file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
    return file

if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) == 2:
        nS = int(args[1])
    else:
        nS = 256
    print('Template Size is '+str(nS))


    filename = GetFileName()
    Temp = cv2.imread(filename,0)
    print('Original Image Size is...')
    print(Temp.shape)
    
    foldname = input('Folder Name? : ')
    MakeDataSet(Temp,nS,'Data/'+foldname)