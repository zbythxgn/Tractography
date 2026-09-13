import numpy as np
import os
import glob
import torch
from scipy.ndimage import map_coordinates
import nibabel as nib
from nibabel import streamlines
from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from dipy.io import read_bvals_bvecs
from nibabel.affines import apply_affine


class VoxCoordDataLoader(object):
    def __init__(self, params):
        self.dwi_path = params['dwi_path']
        self.grad_vecs = params['b_vec']
        self.grad_vals = params['b_val']
        self.brain_mask_path = None
        self.wm_mask_path = params['wm_mask_path']
        self.mode = params['mode']
        assert self.mode in ['train', 'track'], 'mode must be either train or track.'
        if self.mode == 'train':
            self.tractogram_path = params['tractogram_path']
        else:
            self.tractogram_path = None

        self.coords_map = params['coords_map']
        if self.coords_map==True:
            self.registered_dwi = params['registered_dwi']
            self.registered_strmls = params['registered_tractogram']
        else:
            self.registered_dwi = None
            self.registered_strmls = None

        self.max_val = 255
        self.dwi = None
        self.dwi_sh_coeff = None
        self.bvals = None
        self.bvecs = None
        self.brain_mask = np.array([])
        self.wm_mask = np.array([])
        self.arysq = None
        self.affine = None
        self.voxel_sizes = None
        if self.dwi_path is not None:
            self.load_dwi()
            self.load_b_table()

        if self.registered_strmls is not None:
            self.load_strmls_re()

        if self.brain_mask_path is not None:
            self.brain_mask = self.load_mask(self.brain_mask_path)
        if self.wm_mask_path is not None:
            self.wm_mask = self.load_mask(self.wm_mask_path)
        if self.tractogram_path is not None:
            self.load_tractogram()

    def load_dwi(self):
        dwi_file = self.dwi_path
        dwi_data = nib.load(dwi_file)
        self.dwi = dwi_data.get_fdata().astype("float32")
        self.affine = dwi_data.affine.astype("float32")
        self.voxel_sizes = dwi_data.header.get_zooms()[:3]
    
    def load_strmls_re(self):
        re_dwi_data = nib.load(self.registered_dwi)
        self.re_affine = re_dwi_data.affine.astype("float32")
        rasmm2vox_affine1 = np.linalg.inv(self.re_affine)
        self.re_arysq = nib.streamlines.ArraySequence()
        for i in range(len(self.registered_strmls)):
            tractogram_data1 = streamlines.load(self.registered_strmls[i])
            self.re_arysq.extend(tractogram_data1.streamlines)
        self.re_arysq._data = apply_affine(rasmm2vox_affine1,self.re_arysq._data, inplace=True)


    def load_b_table(self):
        bval_file = self.grad_vals
        bvec_file = self.grad_vecs
        self.bvals, self.bvecs = read_bvals_bvecs(bval_file, bvec_file)

    def load_tractogram(self):
        bn_l=[]
        self.arysq = nib.streamlines.ArraySequence()
        for i in range(len(self.tractogram_path)):
            tractogram_data = streamlines.load(self.tractogram_path[i])
            bundle_name=self.tractogram_path[i].split("__")[-1].split(".trk")[0]
            bn_l.extend([bundle_name]*len(tractogram_data.streamlines))
            self.arysq.extend(tractogram_data.streamlines)

        rasmm2vox_affine = np.linalg.inv(self.affine)
        self.arysq._data = apply_affine(rasmm2vox_affine,self.arysq._data, inplace=True)
        self.bundle_name=bn_l

    @staticmethod
    def load_mask(mask_path):
        dwi_data = nib.load(mask_path)
        return dwi_data.get_fdata().astype("float32")

    @staticmethod
    def normalize_dwi(weights, b0):
        """ Normalize dwi by the first b0.
        Parameters:
        -----------
        weights : ndarray of shape (X, Y, Z, #gradients)
            Diffusion weighted images.
        b0 : ndarray of shape (X, Y, Z)
            B0 image.
        Returns
        -------
        ndarray
            Diffusion weights normalized by the B0.
        """
        b0 = b0[..., None]  # Easier to work if it is a 4D array.

        # Make sure in every voxels weights are lower than ones from the b0.
        nb_erroneous_voxels = np.sum(weights > b0)
        if nb_erroneous_voxels != 0:
            print("Nb. erroneous voxels: {}".format(nb_erroneous_voxels))
            weights = np.minimum(weights, b0)

        # Normalize dwi using the b0.
        weights_normed = weights / b0
        weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

        return weights_normed

    def get_spherical_harmonics_coefficients(self, dwi_weights, bvals, bvecs,subname='other', sh_order=8, smooth=0.006, mean_centering=True):
        """ Compute coefficients of the spherical harmonics basis.
        Parameters
        -----------
        dwi_weights : `nibabel.NiftiImage` object
            Diffusion signal as weighted images (4D).
        bvals : ndarray shape (N,)
            B-values used with each direction.
        bvecs : ndarray shape (N, 3)
            Directions of the diffusion signal. Directions are
            assumed to be only on the hemisphere.
        sh_order : int, optional
            SH order. Default: 8
        smooth : float, optional
            Lambda-regularization in the SH fit. Default: 0.006.
        Returns
        -------
        sh_coeffs : ndarray of shape (X, Y, Z, #coeffs)
            Spherical harmonics coefficients at every voxel. The actual number of
            coeffs depends on `sh_order`.
        """

        # Exract the averaged b0.
        # if subname =='hcp':
        #     b0 =5
        # else:
        #     b0=0
        idx1=bvals == 0
        idx2=bvals == 5
        b0_idx = np.logical_or(idx1,idx2)

        b0 = dwi_weights[..., b0_idx].mean(axis=3) + 1e-10

        # Extract diffusion weights and normalize by the b0.
        bvecs = bvecs[np.logical_not(b0_idx)]
        weights = dwi_weights[..., np.logical_not(b0_idx)]
        weights = self.normalize_dwi(weights, b0)

        # Assuming all directions are on the hemisphere.
        # raw_sphere = HemiSphere(xyz=bvecs)
        raw_sphere = Sphere(xyz=bvecs)

        ##raw_sphere.theta.shape=(30,)

        # Fit SH to signal
        sph_harm_basis = sph_harm_lookup.get("descoteaux07")
        Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
        L = -n * (n + 1)
        invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
        data_sh = np.dot(weights, invB.T)
        if mean_centering:
            # Normalization in each direction (zero mean)
            idx = data_sh.sum(axis=-1).nonzero()
            means = data_sh[idx].mean(axis=0)
            data_sh[idx] -= means
        return data_sh

    def resample_dwi(self, directions=None,subname='other', sh_order=8, smooth=0.006, mean_centering=True):
        """ Resamples a diffusion signal according to a set of directions using spherical harmonics.
        Parameters
        -----------
        directions : `dipy.core.sphere.Sphere` object, optional
            Directions the diffusion signal will be resampled to. Directions are
            assumed to be on the whole sphere, not the hemisphere like bvecs.
            If omitted, 100 directions evenly distributed on the sphere will be used.
        sh_order : int, optional
            SH order. Default: 8
        smooth : float, optional
            Lambda-regularization in the SH fit. Default: 0.006.
        """
        data_sh = self.get_spherical_harmonics_coefficients(self.dwi, self.bvals, self.bvecs,subname=subname,
                                                            sh_order=sh_order, smooth=smooth)
        sphere = get_sphere('repulsion100')
        if directions is not None:
            sphere = Sphere(xyz=directions)

        sph_harm_basis = sph_harm_lookup.get("descoteaux07")
        Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
        data_resampled = np.dot(data_sh, Ba.T)
        
        if mean_centering:
            # Normalization in each direction (zero mean)
            idx = data_resampled.sum(axis=-1).nonzero()
            means = data_resampled[idx].mean(axis=0)#(L*W*H,3),(3,)
            data_resampled[idx] -= means
        return data_resampled

    def mask_dwi(self):
        dwi_vol = self.dwi
        mask_vol = self.wm_mask
        if mask_vol.ndim == 3:
            masked_dwi = dwi_vol * np.tile(mask_vol[..., None], (1, 1, 1, dwi_vol.shape[-1]))
        else:
            masked_dwi = dwi_vol * mask_vol

        return masked_dwi

def align_streamlines_to_grid(tract, dwi):
    dwi_vec = np.diag(dwi.affine)[0:3]
    tract_vec = np.diag(tract.affine)[0:3]
    ratio_vec = tract_vec / dwi_vec
    if not all(ratio_vec == 1.0):
        ratio_mat = np.diag(ratio_vec)
        all_streamlines = []
        for i in range(len(tract.streamlines)):
            all_streamlines.append(np.matmul(tract.streamlines[i], ratio_mat))
        return all_streamlines
    else:
        return tract.streamlines

def get_streamlines_lengths(streamlines_list):
    """
INPUT: fibers_list - python list of size N, where each element is a fiber represented by a Ln x 3 np array

OUTPUT: lengths - a np array (vector) of length N, holding the #points in each fiber
"""
    lengths = np.zeros(len(streamlines_list))

    for i in range(len(streamlines_list)):
        lengths[i] = streamlines_list[i].shape[0]

    return lengths

def calc_mean_dwi(dwi, mask):
    DW_means = np.zeros(dwi.shape[3])
    for i in range(len(DW_means)):
        curr_volume = dwi[:, :, :, i]
        if len(mask) > 0:
            curr_volume = curr_volume[mask > 0]
        else:
            curr_volume = curr_volume[curr_volume > 0]
        DW_means[i] = np.mean(curr_volume)

    return DW_means

def get_file_path(target_dir, extension):
    file_path = target_dir + extension
    return file_path