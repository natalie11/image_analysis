# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:03:34 2023

Generate Colour Fractional Anisotropy maps from DWI images
"""

import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io

#Path and filenames
path = 'D:/CRUK_liver/080323/DWI/'
file = 'Xf_b1500_vB1_20230308172628_1701'
output_name = 'CFA.nii'

fimg = os.path.join(path,file+'.nii')
fbval = os.path.join(path,file+'.bval')
fbvec = os.path.join(path,file+'.bvec')

#Load image
img=nib.load(fimg)
data = img.get_fdata()
affine = img.affine #tranformation matrix from voxel space to to world coords (mm)
header = img.header
voxel_size = header.get_zooms()[:3]

#Remove background
S0 = data[:,:,:,0] #Non DW image

from dipy.segment.mask import median_otsu, applymask
S0_mask, mask = median_otsu(S0)
data_mask = applymask(data, mask)
img_mask = nib.Nifti1Image(data_mask, affine)
nib.save(img_mask, os.path.join(path,'img_mask.nii'))

#Gradient table
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

from dipy.reconst.dti import TensorModel, fractional_anisotropy, color_fa
ten_model = TensorModel(gtab)
ten_fit = ten_model.fit(data, mask)
fa = fractional_anisotropy(ten_fit.evals)
cfa = color_fa(fa, ten_fit.evecs)

img_cfa = nib.Nifti1Image(cfa, affine)
nib.save(img_cfa, os.path.join(path,output_name))
