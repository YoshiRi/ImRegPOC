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

########################################### 
# Functrion Test using opencv
###########################################

# Test for POC
import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math
# Library For Evaluation
from POCLibrary import *


# Read image
ref = cv2.imread('../luna1.png',0)
plt.imshow(ref,cmap="gray")

# reference parameter (you can change this)
rdx = 10
rdy = -10
rtheta = 10/180*math.pi
rscale = 1.2
compared = Warp_4dof(ref,rdx,rdy,rtheta,rscale)

# show two image
plt.subplot(1,2,1)
plt.imshow(ref,cmap='gray')
plt.title('reference')
plt.subplot(1,2,2)
plt.imshow(compared,cmap="gray")
plt.title('compared')

plt.show()

# Calclation
Estimated,peak = POC(ref,compared)

# show the results
print("Reference Parameter:")
print(rdx,rdy,rtheta*180/math.pi,rscale)
print("Estimation Results:")
print(Estimated)
print("Peak Value [0-1]:")
print(peak)
