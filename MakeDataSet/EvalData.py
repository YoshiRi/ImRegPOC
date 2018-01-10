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
from FeatureBasedMatching import *

def EvaluationPOC(filename):
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


def EvaluationFP(filename,DES = "SIFT"):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = 50

    # init matcher
    FPmethod = TempMatcher(RefImg,DES)
    Estimation = []
    Peaks = []
    for iterate in range(1,DataNum+1):
        cmpname = filename + 'cmp' + str(iterate) + '.png'
        CmpImg = cv2.imread(cmpname,0)
        param, count, inliner = FPmethod.match(CmpImg)
        Estimation.append(param)
        Peaks.append([count,inliner])
        print(str(iterate)+'image\n')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    FPnum = np.array(Peaks).reshape(DataNum,2)

    np.savetxt(filename + DES + 'Estimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'FPnum'+ DES +'.csv',FPnum,delimiter=',')



if __name__ == '__main__':
    # Get Module
    import os, tkinter, tkinter.filedialog, tkinter.messagebox

    # Show dialog
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("","*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    tkinter.messagebox.showinfo('Choose DataSet Directory','Choose Dataset Directory')
    dirn = tkinter.filedialog.askdirectory(initialdir = iDir)
    dirname = dirn+'/'

    # Do Evaluation
    EvaluationPOC(dirname)
    EvaluationFP(dirname,'SIFT')
    EvaluationFP(dirname,'ORB')