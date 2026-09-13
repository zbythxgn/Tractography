import numpy as np
from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from dipy.core.geometry import sphere_distance
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from scipy.ndimage import map_coordinates
import torch
import nibabel as nib
from nibabel import streamlines
import os
import gc
import subprocess
from fsl.transform.affine import transform


def eval_volume_at_3d_coordinates(volume, coords):
    """ Evaluates the volume data at the given coordinates using trilinear interpolation.
    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    Returns
    -------
    output : 2D array
        Values from volume.
    """
    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(volume, coords.T, order=1, mode="nearest")

    if volume.ndim == 4:
        values_4d = []
        for i in range(volume.shape[-1]):
            values_tmp = map_coordinates(volume[..., i],
                                         coords.T, order=1, mode="nearest")
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)

def get_geometrical_labels(streamlines):
    d_labels = []
    for i in range(len(streamlines)):
        directions = streamlines[i][1:, :] - streamlines[i][0:-1, :]
        directions = directions / np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
        d_labels.append(directions)

    return d_labels

def convert2dwi_patch(dwi_vol,dwi_coord_map, strmls,padl):
    """
    INPUT:  dwi_vol - signal volume 
            strmls  - nib.streamliens.ArraySequence,期望是一个subj所有纤维束,所有点

    """
    #region*/
    #纤维束全部点的坐标，true cube_center
    cube_center = strmls[:-1]#(length-1=t,3)
    ch=dwi_vol.shape[-1]
    valid_l = np.min((len(strmls)-1,padl))#超出设定长度的纤维束，其有效长度就设为padl
    pad_cube=np.zeros((padl,3,3,3,ch))
    pad_targets=np.zeros((padl,3))

    #纤维束方向
    targets=strmls[1:] - strmls[:-1]#(t,3)
    targets = targets / np.sqrt(np.sum(targets**2, axis=1, keepdims=True))
    pad_targets[:valid_l]=targets[:valid_l]



    #由于volume被pad0 1行，z坐标需要统一下移一个单位
    cube_center = cube_center + 4

    orig_cube_center=np.round(cube_center).astype(int)
    orig_cube_center_x=orig_cube_center[:,0]#(t,)
    orig_cube_center_y=orig_cube_center[:,1]#(t,)
    orig_cube_center_z=orig_cube_center[:,2]#(t,)

    #从坐标索引矩阵中取出坐标cube
    orig_cube=[dwi_coord_map[orig_cube_center_x[i]-1:orig_cube_center_x[i]+2,
                        orig_cube_center_y[i]-1:orig_cube_center_y[i]+2,
                        orig_cube_center_z[i]-1:orig_cube_center_z[i]+2] for i in range(len(orig_cube_center)) ]
    orig_cube=np.array(orig_cube)#(t,9,9,9,3)

    
    #计算cube中心点与最近栅格点的各方向偏移
    x_b=cube_center[:,0]-orig_cube_center_x#(t,)
    y_b=cube_center[:,1]-orig_cube_center_y
    z_b=cube_center[:,2]-orig_cube_center_z

    #原cube整体偏移
    a=orig_cube[...,0]+ x_b[:,None,None,None]
    b=orig_cube[...,1]+ y_b[:,None,None,None]
    c=orig_cube[...,2]+ z_b[:,None,None,None]
    cube=np.stack((a,b,c))
    cube=cube.transpose(1,2,3,4,0)#(t,9,9,9,3)
    #endregion*/

    all_pts=cube.reshape(-1,3)#(n_,3)
    dwi_data=eval_volume_at_3d_coordinates(dwi_vol, all_pts)#(n_,ch)
    re_cube=dwi_data.reshape(-1,3,3,3,dwi_vol.shape[-1])#(t,9,9,9,ch)
    pad_cube[:valid_l]=re_cube[:valid_l]
    return pad_cube,pad_targets,np.asarray(valid_l,dtype=int)

