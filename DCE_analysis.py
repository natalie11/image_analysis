# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:06:26 2023

Create a Time To Peak Contrast (TTP) map from Dynamic Contrast Enhanced MRI data, optionally within a labelled Region of Interested
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
from skimage import io

    
def time_to_peak(data, frame=1):
    index = np.where(data == np.amax(data))
    #if multiple points have max value, take lowest
    index = np.amin(index)
    time=index*frame
    return time
   

# Set file path and names of DCE image (.nii) and segmented image (.tif). Set label_file to None if no segmentation is provided   
path = 'D:/CRUK_liver/150223/DCE'
dce_file = 'Xf_DCE_100dyn_cor_20230215180351_1701.nii'
label_file = 'MAX_Xf_DCE_100dyn_cor_20230215180351_1701.labels.tif'
time_res = 16.28

   
# Read first image to get dimensions
fname = os.path.join(path,dce_file)       
dce_img = io.imread(fname)   

dims = dce_img.shape
print(dims)
  
# Fit TTP
ttp = np.zeros([dims[1],dims[2],dims[3]])
for i in trange(dims[1]):
    for j in range(dims[2]):
        for k in range(dims[3]):
            if np.any(dce_img[:,i,j,k]):
                #check for non-zeros
                t = dce_img[:,i,j,k]
                ttp[i,j,k] = time_to_peak(t, frame=time_res)
            else: ttp[i,j,k] = np.nan
            
io.use_plugin('tifffile', 'imsave')
io.imsave(os.path.join(path,'ttp_map.tif'),ttp)

plt.imshow(ttp[10,:,:])
plt.colorbar(label="TTP")
plt.axis('off')
plt.show()
           
# crop vessels
if label_file is not None:
    fname = os.path.join(path,label_file) 
    label_img = io.imread(fname) 
    ttp_crop = ttp
    ttp_crop[label_img==0] = np.nan
    
    io.use_plugin('tifffile', 'imsave')
    io.imsave(os.path.join(path,'ttp_crop.tif'),ttp_crop)
    
    plt.imshow(ttp_crop[15,:,:])
    plt.colorbar(label="TTP")
    plt.axis('off')
    plt.show()