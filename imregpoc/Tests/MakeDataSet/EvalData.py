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
import imregpoc
import fake_imregpoc


global DATANUMBER

def EvalError(Ref,Est):
    err = np.absolute(Ref-Est)
    AVG = np.average(err,axis=0) # vertical sum
    STD = np.std(err,axis=0)
    print(AVG,STD)
    return AVG,STD

def FPErr(Ref,Est,nums,th = 5):
    #print(np.where(peaks>th))
    i,= np.where(nums>th) 
    nRef = Ref[i,:] 
    nEst = Est[i,:]    
    print('Valid Number')
    print(nEst.shape[0])
    AVG,STD = EvalError(nRef,nEst)
    return AVG, STD, nEst.shape[0]/DATANUMBER

def POCErr(Ref,Est,peaks,th = 0.06):
    #print(np.where(peaks>th))
    i,= np.where(peaks>th) 
    nRef = Ref[i,:] 
    nEst = Est[i,:]    
    print('Valid Number')
    print(nEst.shape[0])
    AVG,STD = EvalError(nRef,nEst)
    return AVG, STD, nEst.shape[0]/DATANUMBER

def EvaluationPOC(filename,a=0.5, b=0.8):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = DATANUMBER

    Estimation = []
    Peaks = []
    match = imregpoc.imregpoc(RefImg,RefImg,alpha=a, beta=b)
    for iterate in range(1,DataNum+1):
        cmpname = filename + 'cmp' + str(iterate) + '.png'
        CmpImg = cv2.imread(cmpname,0)
        #param, peak= POC(RefImg,CmpImg)
        match.match_new(CmpImg)
        param = match.getParam()
        param[2] = param[2]/math.pi*180 # rad to deg
        peak = match.getPeak()
        Estimation.append(param)
        Peaks.append(peak)
        print(str(iterate)+'image')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    pocPeaks = np.array(Peaks).reshape(DataNum,1)

    np.savetxt(filename +'POCEstimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'POCpeak.csv',pocPeaks,delimiter=',')
    return 

def EvaluationFP(filename,DES = "SIFT"):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = DATANUMBER

    # init matcher
    FPmethod = imregpoc.TempMatcher(RefImg,DES)
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
    return


def EvaluationPOC_noWindow(filename,a=0.5, b=0.8):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = DATANUMBER

    Estimation = []
    Peaks = []
    match = fake_imregpoc.imregpoc_nowindow(RefImg,RefImg,alpha=a, beta=b)
    #match = fake_imregpoc.imregpoc_noHP(RefImg,RefImg)
    for iterate in range(1,DataNum+1):
        cmpname = filename + 'cmp' + str(iterate) + '.png'
        CmpImg = cv2.imread(cmpname,0)
        #param, peak= POC(RefImg,CmpImg)
        match.match_new(CmpImg)
        param = match.getParam()
        param[2] = param[2]/math.pi*180 # rad to deg        
        peak = match.getPeak()
        Estimation.append(param)
        Peaks.append(peak)
        print(str(iterate)+'image')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    pocPeaks = np.array(Peaks).reshape(DataNum,1)

    np.savetxt(filename +'POC_NW_Estimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'POC_NW_peak.csv',pocPeaks,delimiter=',')
    return

def EvaluationPOC_noLP(filename,a=0.5, b=0.8):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = DATANUMBER

    Estimation = []
    Peaks = []
    match = fake_imregpoc.imregpoc_noLP(RefImg,RefImg,alpha=a)
    for iterate in range(1,DataNum+1):
        cmpname = filename + 'cmp' + str(iterate) + '.png'
        CmpImg = cv2.imread(cmpname,0)
        #param, peak= POC(RefImg,CmpImg)
        match.match_new(CmpImg)
        param = match.getParam()
        param[2] = param[2]/math.pi*180 # rad to deg        
        peak = match.getPeak()
        Estimation.append(param)
        Peaks.append(peak)
        print(str(iterate)+'image')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    pocPeaks = np.array(Peaks).reshape(DataNum,1)

    np.savetxt(filename +'POC_NLP_Estimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'POC_NLP_peak.csv',pocPeaks,delimiter=',')
    return



def EvaluationPOC_noCOG(filename,a=0.5, b=0.8):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = DATANUMBER

    Estimation = []
    Peaks = []
    #match = fake_imregpoc.imregpoc_nowindow(RefImg,RefImg)
    match = fake_imregpoc.imregpoc_noWCOG(RefImg,RefImg,alpha=a, beta=b)
    for iterate in range(1,DataNum+1):
        cmpname = filename + 'cmp' + str(iterate) + '.png'
        CmpImg = cv2.imread(cmpname,0)
        #param, peak= POC(RefImg,CmpImg)
        match.match_new(CmpImg)
        param = match.getParam()
        param[2] = param[2]/math.pi*180 # rad to deg        
        peak = match.getPeak()
        Estimation.append(param)
        Peaks.append(peak)
        print(str(iterate)+'image')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    pocPeaks = np.array(Peaks).reshape(DataNum,1)

    np.savetxt(filename +'POC_NCOG_Estimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'POC_NCOG_peak.csv',pocPeaks,delimiter=',')
    return 


def EvaluationPOC_largeM(filename,a=0.5, b=0.8):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = DATANUMBER

    Estimation = []
    Peaks = []
    #match = fake_imregpoc.imregpoc_nowindow(RefImg,RefImg)
    match = fake_imregpoc.imregpoc_largeM(RefImg,RefImg,alpha=a, beta=b)
    for iterate in range(1,DataNum+1):
        cmpname = filename + 'cmp' + str(iterate) + '.png'
        CmpImg = cv2.imread(cmpname,0)
        #param, peak= POC(RefImg,CmpImg)
        match.match_new(CmpImg)
        param = match.getParam()
        param[2] = param[2]/math.pi*180 # rad to deg        
        peak = match.getPeak()
        Estimation.append(param)
        Peaks.append(peak)
        print(str(iterate)+'image')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    pocPeaks = np.array(Peaks).reshape(DataNum,1)

    np.savetxt(filename +'POC_LM_Estimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'POC_LM_peak.csv',pocPeaks,delimiter=',')
    return 

def EvaluationPOC_NoWhite(filename,a=0.5, b=0.8):
    refname = filename + 'ref.png'
    txtname = filename + 'TrueParam.csv'
    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth

    RefImg = cv2.imread(refname,0)
    DataNum = DATANUMBER

    Estimation = []
    Peaks = []
    #match = fake_imregpoc.imregpoc_nowindow(RefImg,RefImg)
    match = fake_imregpoc.imregpoc_NoWhite(RefImg,RefImg,alpha=a, beta=b)
    for iterate in range(1,DataNum+1):
        cmpname = filename + 'cmp' + str(iterate) + '.png'
        CmpImg = cv2.imread(cmpname,0)
        #param, peak= POC(RefImg,CmpImg)
        match.match_new(CmpImg)
        param = match.getParam()
        param[2] = param[2]/math.pi*180 # rad to deg        
        peak = match.getPeak()
        Estimation.append(param)
        Peaks.append(peak)
        print(str(iterate)+'image')
    Estimated = np.array(Estimation).reshape(DataNum,4)
    pocPeaks = np.array(Peaks).reshape(DataNum,1)

    np.savetxt(filename +'POC_NWi_Estimation.csv',Estimated,delimiter=',')
    np.savetxt(filename +'POC_NWi_peak.csv',pocPeaks,delimiter=',')
    return 


# change contrast a = 15
def ChnageContrast(image,a):
  lut = [ np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)] 
  result_image = np.array( [ lut[value] for value in image.flat], dtype=np.uint8 )
  result_image = result_image.reshape(image.shape)
  return result_image



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


def Comparison_FP(filename):
    txtname = filename + 'TrueParam.csv'

    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth
    POCdata = np.loadtxt(filename +'POCEstimation.csv',delimiter=',')
    POCpeaks = np.loadtxt(filename +'POCpeak.csv',delimiter=',')
    SIFTdata = np.loadtxt(filename + 'SIFT' + 'Estimation.csv',delimiter=',')
    SIFTnum = np.loadtxt(filename + 'FPnum'+ 'SIFT' + '.csv',delimiter=',')
    ORBdata = np.loadtxt(filename + 'ORB' + 'Estimation.csv',delimiter=',')
    ORBnum = np.loadtxt(filename  +'FPnum'+ 'ORB' + '.csv',delimiter=',')
    

    e1 = POCErr(GTdata,POCdata,POCpeaks)
    e2 = FPErr(GTdata,SIFTdata,SIFTnum[:,1])
    e3 = FPErr(GTdata,ORBdata,ORBnum[:,1])

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
    plt.tight_layout()
    plt.show()

    errs = np.array([e1[0],e2[0],e3[0]]).reshape(3,4)
    valids = np.array([e1[2],e2[2],e3[2]]).reshape(3,1)
    # bar graph
    plot_bargraph(errs,['POC','SIFT','ORB'])
    plot_bargraph(valids,['POC','SIFT','ORB'], ["Valid Rate"],['valid rate [%]'])
    plot_bargraph(np.concatenate((valids,errs),axis=1),['POC','SIFT','ORB'], ["Valid Rate","x", "y", "Theta", "Scale"])

def Comparison_POC(filename):
    txtname = filename + 'TrueParam.csv'

    GTdata = np.loadtxt(txtname,delimiter=',') # ground truth
    POCdata = np.loadtxt(filename +'POCEstimation.csv',delimiter=',')
    POCpeaks = np.loadtxt(filename +'POCpeak.csv',delimiter=',')
    POC_NW_data = np.loadtxt(filename +'POC_NLP_Estimation.csv',delimiter=',')
    POC_NW_peaks = np.loadtxt(filename +'POC_NLP_peak.csv',delimiter=',')
    POC_NCOG_data = np.loadtxt(filename +'POC_NCOG_Estimation.csv',delimiter=',')
    POC_NCOG_peaks = np.loadtxt(filename +'POC_NCOG_peak.csv',delimiter=',')
    POC_LM_data = np.loadtxt(filename +'POC_NWi_Estimation.csv',delimiter=',')
    POC_LM_peaks = np.loadtxt(filename +'POC_NWi_peak.csv',delimiter=',')
    

    e1 = POCErr(GTdata,POCdata,POCpeaks)
    e2 = POCErr(GTdata,POC_NW_data,POC_NW_peaks)
    e3 = POCErr(GTdata,POC_NCOG_data,POC_NCOG_peaks)
    e4 = POCErr(GTdata,POC_LM_data,POC_LM_peaks)

    ylim = max(e1[0].max(),e2[0].max(),e3[0].max(),e4[0].max())
    plt.subplot(411)
    plt.bar([1,2,3,4],e1[0])
    plt.title('POC error')
    plt.ylim([0,ylim])
    plt.subplot(412)
    plt.bar([1,2,3,4],e2[0])
    plt.title('No LP error')
    plt.ylim([0,ylim])
    plt.subplot(413)
    plt.bar([1,2,3,4],e3[0])    
    plt.title('No cog')
    plt.ylim([0,ylim])
    plt.subplot(414)
    plt.bar([1,2,3,4],e4[0])    
    plt.title('No white')
    plt.ylim([0,ylim])
    plt.tight_layout()
    plt.show()


def plot_bargraph(data,legend_labels=["A", "B", "C"],xlabels= ["x", "y", "Theta", "Scale"], ylabels = 'Error [pix,deg]'):
    num_sample = data.shape[0]
    num_item_per_sample = data.shape[1]

    ## 凡例用のラベル（サンプル数だけ必要）
    ## 凡例の数だけ色を用意する
    colors = ["red", "green", "blue"]
    ## 棒グラフの幅
    width = 0.25
    ## 余白
    margin = 0.2
    ## 1尺度あたりのデータを並べるのに必要な幅。
    block = width * num_sample + margin
    ## 棒グラフ（長方形）の左下の位置の基準にするポイント
    ind = np.arange(num_item_per_sample) * block
    ## 各サンプルについて、棒グラフを描画する
    for i in range(num_sample):
        plt.bar(
            ind + width*i, 		## 棒グラフの左下の点の座標。データ毎に少しずつズラす
            data[i], 			## 各始点にプロットされる1次元配列
            width, 				## 棒の幅
            color=colors[i], 	## 棒の色
            label=legend_labels[i]	 	## 棒の凡例名
            )
    ##x軸にラベルを表示する位置を設定する。
    xlocs = ind + width * num_sample / 2.
    ## xtics(labelの位置, label), labelは1次元配列
    plt.xticks(xlocs, xlabels )
    ## 余白を加味したx軸方向の変域
    plt.xlim(-margin, ind[-1]+width*num_sample+margin)
    ## Y軸ラベル
    plt.ylabel(ylabels)
    ## 凡例を表示
    plt.legend(prop={'size' : 18},loc="upper right")
    ## グリッドを表示
    #plt.grid(True)
    plt.show()



if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) == 2:
        dirname = args[1]
    else:
        dirname = getDir()
    
    DATANUMBER = 100
    
    ## Do Evaluation
    # poc tuning
    # alpha = 0.5     beta = 0.8
    alpha = 0.5
    beta = 0.6

    # EvaluationPOC(dirname,alpha,beta)
    #  # EvaluationPOC_noWindow(dirname)
    EvaluationPOC_noLP(dirname)
    # EvaluationPOC_noCOG(dirname,alpha,beta)
    # EvaluationPOC_largeM(dirname,alpha,beta)
    EvaluationPOC_NoWhite(dirname,alpha,beta)
    # EvaluationFP(dirname,'SIFT')
    # EvaluationFP(dirname,'ORB')

    #Comparison_FP(dirname)
    Comparison_POC(dirname)