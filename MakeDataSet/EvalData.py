
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


def Evaluation(filename):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = 50

    Estimation = []
    Peaks = []
    for iterate in range(1,DataNum+1):
        cmpname = filename + 'cmp' + str(iterate) + '.png'
        CmpImg = cv2.imread(cmpname,0)
        param, peak= POC(RefImg,CmpImg)
        Estimation.append(param)
        Peaks.append(peak)
        print(str(iterate)+'image\n')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    pocPeaks = np.array(Peaks).reshape(DataNum,1)

    np.savetxt(filename +'POCEstimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'POCpeak.csv',pocPeaks,delimiter=',')


if __name__ == '__main__':
    filename = 'Data/test1'
    Evaluation(filename)