# Robust FFT-Based Image Registration Tools for Python
This program provides robust image registration method using "Phase Correlation" technique.

With this toolbox, you can estimate 
Translation, Rotation and Scaling between two images.

Output form is supported: Affine Matrix, each parameter
If you used this programs, please cite our paper below:

>  Y. Ri, H. Hiroshi Fujimoto. : \`\`Practical Phase-Only Correlation Algorithm for Robust and Accurate Image Sencing'', <i> Now Under Review </i> Vol. X, pp.XX. (20XX) 

## Getting started 
### Prerequirements
This package requires python3.X and opencv3.X and other related sources.

For windows and Linux users, anaconda is one of the choise to install both of this.

### Instaling


## Running the tests


# Short description
## 1. Phase-Correlation based Translation Estimate (POC)
Image translation can be detected with the cross correlation of 2D FFT spectrum of spacial frequency.

$$
\begin{eqnarray}
 R(k_1,k_2)=\frac{F(k_1,k_2)\overline{G(k_1,k_2)}}{|F(k_1,k_2)\overline{G(k_1,k_2)}|}
\end{eqnarray}
$$

## 2. Phase-Correlation based Rotation and Scaling Estimate (RIPOC)
Using log-polar trasformation, rotation and scaling can also detected with phase correlation technique.


$$
\begin{eqnarray}
	(\delta_x,\delta_y)=(N\theta/\pi,-N\log_N \kappa)
\end{eqnarray}
$$

## Other related links

>  K. Takita, T. Aoki, Y. Sasaki, T. Higuchi and K. Kobayashi. : \`\`High-Accuracy Subpixel Image Registration Based on Phase-Only Correlation'', <i> IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences </i> Vol. E86-A, pp. 1925-1934. (2003) 
