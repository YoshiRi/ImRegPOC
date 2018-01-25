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


def EvalError(Ref,Est):
    err = np.absolute(Ref-Est)
    AVG = np.average(err,axis=0) # vertical sum
    STD = np.std(err,axis=0)
    print(AVG,STD)
    return AVG,STD

def POCErr(Ref,Est,peaks):
    th = 0.02
    #print(np.where(peaks>th))
    i,= np.where(peaks>th) 
    nRef = Ref[i,:] 
    nEst = Est[i,:]    
    print('Valid Number')
    print(nEst.shape[0])
    AVG,STD = EvalError(nRef,nEst)
    return AVG, STD

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
        print(str(iterate)+'image')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    pocPeaks = np.array(Peaks).reshape(DataNum,1)

    np.savetxt(filename +'POCEstimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'POCpeak.csv',pocPeaks,delimiter=',')
    return POCErr(GTdata,Estimated,pocPeaks)

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
        print(str(iterate)+'image')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    FPnum = np.array(Peaks).reshape(DataNum,2)

    np.savetxt(filename + DES + 'Estimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'FPnum'+ DES +'.csv',FPnum,delimiter=',')
    return EvalError(GTdata,Estimated)

def getDir():
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
    return dirname


def Comparison(filename):
    txtname = filename + 'TrueParam.csv'

    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth
    POCdata = np.loadtxt(filename +'POCEstimation.csv',delimiter=',')
    POCpeaks = np.loadtxt(filename +'POCpeak.csv',delimiter=',')
    SIFTdata = np.loadtxt(filename + 'SIFT' + 'Estimation.csv',delimiter=',')
    ORBdata = np.loadtxt(filename + 'ORB' + 'Estimation.csv',delimiter=',')

    e1 = POCErr(GTdata,POCdata,POCpeaks)
    e2 = EvalError(GTdata,SIFTdata)
    e3 = EvalError(GTdata,ORBdata)

    ylim = max(e1[0].max(),e2[0].max(),e3[0].max())
    plt.subplot(311)
    plt.bar([1,2,3,4],e1[0])
    plt.title('POC error')
    plt.ylim([0,ylim])
    plt.subplot(312)
    plt.bar([1,2,3,4],e2[0])
    plt.title('SIFT error')
    plt.ylim([0,ylim])
    plt.subplot(313)
    plt.bar([1,2,3,4],e3[0])    
    plt.title('ORB error')
    plt.ylim([0,ylim])
    plt.show()

if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) == 2:
        dirname = args[1]
    else:
        dirname = getDir()

    # Do Evaluation
    #EvaluationPOC(dirname)
    #EvaluationFP(dirname,'SIFT')
    #EvaluationFP(dirname,'ORB')

    Comparison(dirname)
