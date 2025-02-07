import os
import glob
import time
import argparse

import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from ..DeepStrain.data.nifti_dataset import resample_nifti
from ..DeepStrain.data import base_dataset
from ..DeepStrain.data.base_dataset import _roll2center_crop
from scipy.ndimage.measurements import center_of_mass

from tensorflow.keras.optimizers import Adam
from ..DeepStrain.options.test_options import TestOptions
from ..DeepStrain.models import deep_strain_model
from ..DeepStrain.utils import myocardial_strain
from scipy.ndimage import gaussian_filter

def normalize(x, axis=(0,1,2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x-mu)/(sd+1e-8)

def get_mask(V):
    nx, ny, nz, nt = V.shape
    
    M = np.zeros((nx,ny,nz,nt))
    v = V.transpose((2,3,0,1)).reshape((-1,nx,ny)) # (nz*nt,nx,ny)
    v = normalize(v)
    m = netS(v[:,nx//2-64:nx//2+64,ny//2-64:ny//2+64,None])
    M[nx//2-64:nx//2+64,ny//2-64:ny//2+64] += np.argmax(m, -1).transpose((1,2,0)).reshape((128,128,nz,nt))
    
    return M

class Options():
    
    def __init__(self):
        self.isTrain = False
        self.image_shape = (128,128,1)
        self.volume_shape = (128,128,16,1)
        self.nlabels = 4
        self.pretrained_models_netS  = 'DeepStrain/pretrained_models/carson_Jan2021.h5'
        self.pretrained_models_netME = 'DeepStrain/pretrained_models/carmen_Jan2021.h5'
        
opt = Options()

model = deep_strain_model.DeepStrain(Adam, opt=opt)
netS  = model.get_netS()
netME = model.get_netME()

## Parameters of the script
parser = argparse.ArgumentParser(description="Run DeepStrain processing on given datasets.")
parser.add_argument("--datadirs", nargs="+", required=True, help="List of dataset directories.")
parser.add_argument("--out_folder", required=True, help="Output folder for results.")

args = parser.parse_args()
datadirs = args.datadirs
out_folder = args.out_folder

# datadirs = ['dataset/ACDC/C','dataset/ACDC/MCD', 'dataset/LaFe/C', 'dataset/LaFe/MCA', 'dataset/LaFe/MCD', 'dataset/LaFe/MU']
# out_folder = "results1"

os.makedirs(out_folder, exist_ok=True)

## Get 4d segmentation
for datadir in datadirs:
    for image_path in glob.glob(f'{datadir}/*'):
        V_nifti = nib.load(image_path)
        V_nifti_resampled = resample_nifti(V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=None)

        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        name_without_ext = os.path.splitext(name_without_ext)[0]
        
        # here we normalize per image, not volume
        V = V_nifti_resampled.get_fdata()
        V = normalize(V, axis=(0,1))
        
        # In this case we don't yet have a segmentation we can use to crop the image. 
        # In most cases we can simply center crop (see `get_mask` function): 
        M = get_mask(V)
        
        # However, in some instances the image could end up outside the FOV. While this is unlikely, please inspect 
        # your data.
        
        # We could use VCN to center the heart, but an alternative approach is to use V as an approximate segmentation
        # to center the image. This of course means that we need to do the inverse opereation if we want to recover 
        # the image as a nifti. 

        # get approximate segmentation (in most cases this is enough)
        M = get_mask(V)

        # now we calculate center of mass using the first frame as reference as before.
        center_resampled = center_of_mass(M[:,:,:,0]==2)
        V = base_dataset.roll_and_pad_256x256_to_center(x=V, center=center_resampled)
        M = base_dataset.roll_and_pad_256x256_to_center(x=M, center=center_resampled)
        center_resampled_256x256 = center_of_mass(M==3)

        # we save all this info to invert the segmentation bask to its original location/resolution
        nifti_info = {'affine'         : V_nifti.affine,
                    'affine_resampled' : V_nifti_resampled.affine,
                    'zooms'            : V_nifti.header.get_zooms(),
                    'zooms_resampled'  : V_nifti_resampled.header.get_zooms(),
                    'shape'            : V_nifti.shape,
                    'shape_resampled'  : V_nifti_resampled.shape,
                    'center_resampled' : center_resampled,
                    'center_resampled_256x256' : center_resampled_256x256} 

        M = get_mask(V)[128-64:128+64,128-64:128+64]
        M_nifti = base_dataset.convert_back_to_nifti(M, nifti_info, inv_256x256=True, order=1, mode='nearest')

        # easy. Let's save the results now. 
        os.makedirs(os.path.join(out_folder, datadir), exist_ok=True)
        M_nifti.to_filename(os.path.join(out_folder, datadir, f'{name_without_ext}.nii.gz'))


## Motion estimation images
for datadir in datadirs:
    for image_path, motion_path in zip(glob.glob(f'{datadir}/*'),
            glob.glob(f'{out_folder}/{datadir}/*')):
        ES = 16

        V_nifti = nib.load(image_path)
        M_nifti = nib.load(motion_path)

        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        name_without_ext = os.path.splitext(name_without_ext)[0]

        V_nifti_resampled = resample_nifti(V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16)
        M_nifti_resampled = resample_nifti(M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16)

        center = center_of_mass(M_nifti_resampled.get_fdata()[:,:,:,0]==2)
        V = _roll2center_crop(x=V_nifti_resampled.get_fdata(), center=center)
        M = _roll2center_crop(x=M_nifti_resampled.get_fdata(), center=center)

        # model was trained with cine data from base to apex.
        I = np.argmax((M==1).sum(axis=(0,1,3)))
        if I > M.shape[2]//2:
            print('Apex to Base. Inverting.')
            V = V[:,:,::-1]
            M = M[:,:,::-1]

        V = normalize(V, axis=(0,1,2))
        
        V_0 = V[...,0][None,...,None]
        V_t = V[...,ES][None,...,None]

        y_t = netME([V_0, V_t]).numpy()
        y_t = gaussian_filter(y_t, sigma=(0,2,2,0,0))
            
        mask_end_diastole = M[..., 0]
        
        strain = myocardial_strain.MyocardialStrain(mask=mask_end_diastole, flow=y_t[0,:,:,:,:])
        strain.calculate_strain(lv_label=2)

        base_im_dir = os.path.join(out_folder, "images")

        im_dir = os.path.join(base_im_dir, datadir, name_without_ext)
        circumferential_dir = os.path.join(im_dir, "circumferential")
        radial_dir = os.path.join(im_dir, "radial")

        os.makedirs(circumferential_dir, exist_ok=True)
        os.makedirs(radial_dir, exist_ok=True)

        for i in range(strain.Ecc.shape[-1]):
            filename = os.path.join(circumferential_dir, f"Ecc_{i:03d}.png")
            plt.imsave(filename, strain.Ecc[..., i], cmap='RdBu_r')

        for i in range(strain.Err.shape[-1]):
            filename = os.path.join(radial_dir, f"Err_{i:03d}.png")
            plt.imsave(filename, strain.Err[..., i], cmap='RdBu')