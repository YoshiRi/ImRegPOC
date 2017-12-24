# Test for POC
import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlibの描画系
import math
# Library For Evaluation
from POCLibrary import *


# Read image
ref = cv2.imread('luna1.png',0)
plt.imshow(ref,cmap="gray")

# reference parameter
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

# Evaluation
# transfer 
ref = np.float64(ref)
compared = np.float64(compared)

# Look at result
POC(ref,compared)
plt.show()