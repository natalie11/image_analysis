# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:06:26 2023

Simple fitting of DWI data to calculate ACD from 2 b-values
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
    
def fit_twopoint(b, y, C=0):
    y = y - C
    ADC = -(np.log(y[0]/y[1]))/(b[0]-b[1])
    return ADC
   

# Set file path and names of 2 diffusion weighted images with different b values    
path = 'F:/CRUK_liver/230118/Nifti'
files =  ['DICOM_b90_vB1_20230118174836.nii',          
          'DICOM_b2000_vB1_20230118174836.nii',
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
for i in trange(dims[0]):
    for j in range(dims[1]):
        for k in range(dims[2]):
            # Non-linear Fit
            y = np.sum(data[:,i,j,k,:3],axis=-1)
            if y[0]>10 and not np.any(y==0):
                b = bv[:,0]
                ADC = fit_twopoint(b, y)
                if True:
                    adc[i,j,k] = ADC


adc[adc<0] = 0.
adc_img = nib.Nifti1Image(adc, tr)
nib.save(adc_img, os.path.join(path,output_name))

plt.imshow(adc[:,:,8],vmin=0,vmax=1e-3)
plt.colorbar(label="ADC")
plt.axis('off')
plt.show()