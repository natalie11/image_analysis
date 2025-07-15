# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:06:26 2023

Simple fitting of DWI data to calculate ACD
"""

import numpy as np
arr = np.asarray
import nibabel as nib
import os
from matplotlib import pyplot as plt
import scipy as sp
import scipy.optimize
from tqdm import tqdm, trange
import json

def model_func(t, A, K, C):
    return A * np.exp(-K * t) + C
    
def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(-A_log)
    return A, K

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=1000)
    A, K, C = opt_parms
    return A, K, C    

# Set file path and names of >= 2 diffusion weighted images with different b values     
path = 'D:/CRUK_liver/150223/DWI/'
files =  ['Xf_b500_vB1_20230215180351_1001-resample.nii', 
          'Xf_b90_vB1_20230215180351_901.nii',
          'Xf_SWITCH_DB_TO_YES_b3000_80_20230215180351_1101-resample.nii'
          ]
output_name = 'ADC.nii'

# Grab header information
hdr_files = [f.replace('.nii','.json') for f in files]
hdr = []
for f in hdr_files:
    with open(os.path.join(path,f),'r') as fo:
        hdr.append(json.load(fo))
          
# Grab b-value data          
bval_files = [f.replace('.nii','.bval') for f in files]
bv = []
for f in bval_files:
    with open(os.path.join(path,f),'r') as fo:
        bvcur = [float(x) for x in fo.read().strip().split(' ')]
        bv.append(bvcur)
bv = arr(bv) 
   
# Read first image to get dimensions
fname = os.path.join(path,files[0])       
img = nib.load(fname)   
tr = img.affine
data = img.get_fdata()
dims = list(data.shape)

# Read all data
data = np.zeros([len(files)]+dims,dtype=data.dtype)
for i,f in enumerate(files):
    print(i,f)
    img = nib.load(os.path.join(path,f))   
    data[i] = img.get_fdata() / hdr[i]['PhilipsRescaleSlope']

# Fit ADC
adc = np.zeros([dims[0],dims[1],dims[2]])
S0 = np.zeros([dims[0],dims[1],dims[2]])
for i in trange(dims[0]):
    for j in range(dims[1]):
        for k in range(dims[2]):
            y = np.sum(data[:,i,j,k,:3],axis=-1)
            if y[0]>10 and not np.any(y==0):
                b = bv[:,0]
                A, K = fit_exp_linear(b, y)
                adc[i,j,k] = -K
                S0[i,j,k] = A


adc[adc<0] = 0.
adc_img = nib.Nifti1Image(adc, tr)
nib.save(adc_img, os.path.join(path,output_name))

plt.imshow(adc[:,:,8],vmin=0,vmax=1.5e-3)
plt.colorbar(label="ADC")
plt.axis('off')
plt.show()