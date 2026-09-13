import numpy as np
from scipy.ndimage import map_coordinates
from train_utils import eval_volume_at_3d_coordinates
import torch
import math
from tqdm import tqdm
from nibabel import streamlines
from time import time
import sys
from dipy.core.sphere import Sphere, HemiSphere
from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
import os
import gc
import subprocess
# from fsl.transform.affine import transform


def calc_angles_matrix(sphere):
    """
INPUT: sphere object with d vertices uniformally distributed on the unit sphere

OUTPUT: theta_mat = dxd symmetric matrix, where theta_mat[i,j]=angle(vi,vj) in degrees
# """
    vecs = np.zeros((len(sphere.x), 3))
    vecs[:, 0] = sphere.x
    vecs[:, 1] = sphere.y
    vecs[:, 2] = sphere.z

    dot_mat = np.matmul(vecs, vecs.T)
    norm_vec = np.linalg.norm(vecs, axis=1)[:, None]
    norm_mat = np.matmul(norm_vec, norm_vec.T)

    res_mat = np.clip(np.divide(dot_mat, norm_mat), -1, 1)
    theta_mat = np.array(np.arccos(res_mat))
    theta_mat = np.nan_to_num(theta_mat)
    theta_mat_deg = np.degrees(theta_mat)

    return theta_mat_deg

def idx2direction(direction_idxs, sphere):
    """
INPUT: direction_idxs - tensor of size N x 1 (where N is the batch size) holding indices of sphere directions

OUTPUT: direction_vecs - tensor of size N x 3 (where N is the batch size) holding corresponding sphere directions
"""
    direction_vecs = np.zeros((len(direction_idxs), 3))

    for i in range(len(direction_idxs)):
        if direction_idxs[i] == 724:
            direction_vecs[i, :] = [0.0, 0.0, 0.0]
        else:
            a = sphere.x[direction_idxs[i]]
            b = sphere.y[direction_idxs[i]]
            c = sphere.z[direction_idxs[i]]
            d = np.concatenate((a,b,c),axis=0)
            direction_vecs[i, :] = d
    return direction_vecs

def make_is_outside_mask(mask, positions,affine=None, threshold=0.05):
    """ Makes a function that checks which streamlines have their last coordinates outside a mask.

    Parameters
    ----------
    mask : 3D array
        3D image defining a mask. The interior of the mask is defined by voxels with value higher or equal to `threshold`.
    affine : ndarray of shape (4, 4)
        Matrix representing the affine transformation that aligns streamlines coordinates on top of `mask`.
    threshold : float
        Voxels value higher or equal to this threshold are considered as part of the interior of the mask.

    Returns
    -------
    function
    """
    mask_values = map_coordinates_3d_4d(mask, positions, affine, order=1)
    return mask_values < threshold

def map_coordinates_3d_4d(input_array, indices, affine=None, order=1,displacement=0):
    """ Evaluate the input_array data at the given indices
    using trilinear interpolation

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array

    Notes
    -----
    At some point this will be merged in Dipy. See PR #587.
    """
    if affine is not None:
        inv_affine = np.linalg.inv(affine)
        indices -= displacement
        indices = (np.dot(indices, inv_affine[:3, :3]) + inv_affine[:3, 3])
        

    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=order)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=order)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)

def calc_angles(direc, direc_previous,device):
    a = direc * direc_previous
    inner_product = a.sum(1).to(device)
    fanshu_direc = (direc * direc).sum(1).to(device)
    fanshu_direc_prev = (direc_previous * direc_previous).sum(1).to(device)
    # norm_length = torch.ones(len(direc)).to(device)
    # cos = inner_product/(math.sqrt(fanshu_direc)* math.sqrt(fanshu_direc_prev))
    cos = torch.tensor([inner_product[i]/(math.sqrt(fanshu_direc[i])* math.sqrt(fanshu_direc_prev[i])) for i in range(len(inner_product))]).to(device)
    pi = math.pi
    degrees = 180*(torch.acos(cos))/pi
    degrees = degrees.to(device)
    return degrees

def output_tractogram(streamlines_list):
    tractogram_array = streamlines.array_sequence.ArraySequence()
    for j in tqdm(range(len(streamlines_list))):
        for i in range(len(streamlines_list[j])):
            streamline = streamlines_list[j][i].detach().cpu().numpy()
            tractogram_array.append(streamline)
        print("saving streamline batch number ", j, ' out of ', len(streamlines_list))
    # for j in tqdm(range(len(streamlines_list))):
    #     streamline = streamlines_list[j].detach().cpu().numpy()
    #     tractogram_array.append(streamline)
    #     print("saving streamline count number ", j, ' out of ', len(streamlines_list))

    return tractogram_array

def fiber_lengths(fibers_list, voxel_size, device):
    """
    INPUT: fibers_list - python list of size N, where each element is a fiber represented by a Ln x 3 np array
            voxel_size - a vector of size 3, (vx,vy,vz), representing the voxel size in mm.

    OUTPUT: lengths - a np array (vector) of length N, holding the tota, arc-length of each fiber
    """
    lengths = torch.zeros(len(fibers_list))

    for i in range(len(fibers_list)):
        a = fibers_list[i][1:, :].to(device).to(torch.float32)
        b = fibers_list[i][:-1, :].to(device).to(torch.float32)
        single_step_len = ((a - b) * voxel_size).to(torch.float32)
        lengths[i] = sum(torch.linalg.norm(single_step_len, axis=1))

    return lengths


class Timer():
    """ Times code within a `with` statement. """
    def __init__(self, txt, newline=False):
        self.txt = txt
        self.newline = newline

    def __enter__(self):
        self.start = time()
        if not self.newline:
            print(self.txt + "... ", end="")
            sys.stdout.flush()
        else:
            print(self.txt + "... ")

    def __exit__(self, type, value, tb):
        if self.newline:
            print(self.txt + " done in ", end="")

        print("{:.2f} sec.".format(time()-self.start))


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
    # Should not happen, but with the noise we never know!
    nb_erroneous_voxels = np.sum(weights > b0)
    if nb_erroneous_voxels != 0:
        print ("Nb. erroneous voxels: {}".format(nb_erroneous_voxels))
        weights = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

    return weights_normed



#for l2t_numpy_based tracking
def get_spherical_harmonics_coefficients(dwi, bvals, bvecs, sh_order=8, smooth=0.006, first=False, mean_centering=True):
    """ Compute coefficients of the spherical harmonics basis.

    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
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
    mean_centering : bool
        If True, signal will have zero mean in each direction for all nonzero voxels

    Returns
    -------
    sh_coeffs : ndarray of shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number of
        coeffs depends on `sh_order`.
    """
    bvals = np.asarray(bvals)
    bvecs = np.asarray(bvecs)
    dwi_weights = dwi.get_fdata().astype("float32")

    # Exract the averaged b0.
    b0_idx = bvals == 0
    b0 = dwi_weights[..., b0_idx].mean(axis=3)

    # Extract diffusion weights and normalize by the b0.
    bvecs = bvecs[np.logical_not(b0_idx)]
    weights = dwi_weights[..., np.logical_not(b0_idx)]
    weights = normalize_dwi(weights, b0)

    # Assuming all directions are on the hemisphere.
    raw_sphere = HemiSphere(xyz=bvecs)

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


"""
this function gets the 3*3*3(voxel length) pathch of arbitrary points.
But very slow
"""


def get_patchdata_v2(dwi_vol,seeds):

#z=0，9
#region/
    #取邻域坐标，非取整，而是333vox
    coords110 = np.zeros((len(seeds),3))
    coords110[:,0] = seeds[:,0]
    coords110[:,1] = seeds[:,1]
    coords110[:,2] = seeds[:,2]-1


    coords100 = np.zeros((len(seeds),3))
    coords100[:,0] = seeds[:,0]
    coords100[:,1] = seeds[:,1]-1
    coords100[:,2] = seeds[:,2]-1


    coords120 = np.zeros((len(seeds),3))
    coords120[:,0] = seeds[:,0]
    coords120[:,1] = seeds[:,1]+1
    coords120[:,2] = seeds[:,2]-1


    coords010 = np.zeros((len(seeds),3))
    coords010[:,0] = seeds[:,0]-1
    coords010[:,1] = seeds[:,1]
    coords010[:,2] = seeds[:,2]-1


    coords000 = np.zeros((len(seeds),3))
    coords000[:,0] = seeds[:,0]-1
    coords000[:,1] = seeds[:,1]-1
    coords000[:,2] = seeds[:,2]-1


    coords020 = np.zeros((len(seeds),3))
    coords020[:,0] = seeds[:,0]-1
    coords020[:,1] = seeds[:,1]+1
    coords020[:,2] = seeds[:,2]-1



    coords220 = np.zeros((len(seeds),3))
    coords220[:,0] = seeds[:,0]+1
    coords220[:,1] = seeds[:,1]+1
    coords220[:,2] = seeds[:,2]-1



    coords210 = np.zeros((len(seeds),3))
    coords210[:,0] = seeds[:,0]+1
    coords210[:,1] = seeds[:,1]
    coords210[:,2] = seeds[:,2]-1



    coords200 = np.zeros((len(seeds),3))
    coords200[:,0] = seeds[:,0]+1
    coords200[:,1] = seeds[:,1]-1
    coords200[:,2] = seeds[:,2]-1

#endregion/

#z=1，8
#region*/
    coords101 = np.zeros((len(seeds),3))
    coords101[:,0] = seeds[:,0]
    coords101[:,1] = seeds[:,1]-1
    coords101[:,2] = seeds[:,2]


    coords121 = np.zeros((len(seeds),3))
    coords121[:,0] = seeds[:,0]
    coords121[:,1] = seeds[:,1]+1
    coords121[:,2] = seeds[:,2]

              
    coords011 = np.zeros((len(seeds),3))
    coords011[:,0] = seeds[:,0]-1
    coords011[:,1] = seeds[:,1]
    coords011[:,2] = seeds[:,2]


    coords001 = np.zeros((len(seeds),3))
    coords001[:,0] = seeds[:,0]-1
    coords001[:,1] = seeds[:,1]-1
    coords001[:,2] = seeds[:,2]



    coords021 = np.zeros((len(seeds),3))
    coords021[:,0] = seeds[:,0]-1
    coords021[:,1] = seeds[:,1]+1
    coords021[:,2] = seeds[:,2]

    coords221 = np.zeros((len(seeds),3))
    coords221[:,0] = seeds[:,0]+1
    coords221[:,1] = seeds[:,1]+1
    coords221[:,2] = seeds[:,2]


    coords211 = np.zeros((len(seeds),3))
    coords211[:,0] = seeds[:,0]+1
    coords211[:,1] = seeds[:,1]
    coords211[:,2] = seeds[:,2]


    coords201 = np.zeros((len(seeds),3))
    coords201[:,0] = seeds[:,0]+1
    coords201[:,1] = seeds[:,1]-1
    coords201[:,2] = seeds[:,2]

#endregion*/

#z=2，9
#region*/
    coords112 = np.zeros((len(seeds),3))
    coords112[:,0] = seeds[:,0]
    coords112[:,1] = seeds[:,1]
    coords112[:,2] = seeds[:,2]+1


    coords102 = np.zeros((len(seeds),3))
    coords102[:,0] = seeds[:,0]
    coords102[:,1] = seeds[:,1]-1
    coords102[:,2] = seeds[:,2]+1



    coords122 = np.zeros((len(seeds),3))
    coords122[:,0] = seeds[:,0]
    coords122[:,1] = seeds[:,1]+1
    coords122[:,2] = seeds[:,2]+1



    coords012 = np.zeros((len(seeds),3))
    coords012[:,0] = seeds[:,0]-1
    coords012[:,1] = seeds[:,1]
    coords012[:,2] = seeds[:,2]+1


    coords002 = np.zeros((len(seeds),3))
    coords002[:,0] = seeds[:,0]-1
    coords002[:,1] = seeds[:,1]-1
    coords002[:,2] = seeds[:,2]+1


    coords022 = np.zeros((len(seeds),3))
    coords022[:,0] = seeds[:,0]-1
    coords022[:,1] = seeds[:,1]+1
    coords022[:,2] = seeds[:,2]+1

    coords222 = np.zeros((len(seeds),3))
    coords222[:,0] = seeds[:,0]+1
    coords222[:,1] = seeds[:,1]+1
    coords222[:,2] = seeds[:,2]+1


    coords212 = np.zeros((len(seeds),3))
    coords212[:,0] = seeds[:,0]+1
    coords212[:,1] = seeds[:,1]
    coords212[:,2] = seeds[:,2]+1

    coords202 = np.zeros((len(seeds),3))
    coords202[:,0] = seeds[:,0]+1
    coords202[:,1] = seeds[:,1]-1
    coords202[:,2] = seeds[:,2]+1

    #endregion*/

    all_coords= np.concatenate((coords000,coords001,coords002,coords010,coords011,coords012,coords020,coords021,coords022,
                coords100,coords101,coords102,coords110,seeds,coords112,coords120,coords121,coords122,
                coords200,coords201,coords202,coords210,coords211,coords212,coords220,coords221,coords222),axis=0)#(27*all_pts,3)

    all_dwi= eval_volume_at_3d_coordinates(dwi_vol, all_coords)#(n,45) 
    orig_dir='/home/yangyiqiong/HCP_proc/neighbour_coord_'+'.txt'
    regis_dir='/home/yangyiqiong/HCP_proc/regis_coord_'+'.txt'
    np.savetxt(orig_dir,all_coords,fmt='%0.8f')
    os.environ['orig_dir']=str(orig_dir)  #environ的键值必须是字符串
    os.environ['regis_dir']=str(regis_dir)
    subprocess.call("cat $orig_dir | img2imgcoord -src '/home/yangyiqiong/eval_pipeline/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/NoArtifacts_Relaxation.nii.gz' -dest '/home/yangyiqiong/eval_pipeline/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/registored_all.nii.gz' -xfm '/home/yangyiqiong/eval_pipeline/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/registor_affine.mat' - > $regis_dir",shell=True)
    subprocess.call("sed -i '1d' $regis_dir",shell=True)
    all_coords_maps =np.loadtxt(regis_dir)
    all_data=np.concatenate((all_dwi,all_coords_maps),axis=-1)#(27*all_pts,48)
    os.remove(orig_dir)
    os.remove(regis_dir)
    
    a = np.zeros((27,len(seeds),all_data.shape[-1]))
    for i in range(27):
        start=i*int(len(seeds))
        end = (i+1)*int(len(seeds))
        a[i]=all_data[start:end,:]
    a = a.transpose(1,0,2)
    a = a.reshape(len(seeds),3,3,3,all_data.shape[-1])
    return a


def get_patchdata_v3(dwi_vol,seeds):

#z=0，9
#region/
    #取邻域坐标，非取整，而是333vox
    coords110 = np.zeros((len(seeds),3))
    coords110[:,0] = seeds[:,0]
    coords110[:,1] = seeds[:,1]
    coords110[:,2] = seeds[:,2]-1


    coords100 = np.zeros((len(seeds),3))
    coords100[:,0] = seeds[:,0]
    coords100[:,1] = seeds[:,1]-1
    coords100[:,2] = seeds[:,2]-1


    coords120 = np.zeros((len(seeds),3))
    coords120[:,0] = seeds[:,0]
    coords120[:,1] = seeds[:,1]+1
    coords120[:,2] = seeds[:,2]-1


    coords010 = np.zeros((len(seeds),3))
    coords010[:,0] = seeds[:,0]-1
    coords010[:,1] = seeds[:,1]
    coords010[:,2] = seeds[:,2]-1


    coords000 = np.zeros((len(seeds),3))
    coords000[:,0] = seeds[:,0]-1
    coords000[:,1] = seeds[:,1]-1
    coords000[:,2] = seeds[:,2]-1


    coords020 = np.zeros((len(seeds),3))
    coords020[:,0] = seeds[:,0]-1
    coords020[:,1] = seeds[:,1]+1
    coords020[:,2] = seeds[:,2]-1



    coords220 = np.zeros((len(seeds),3))
    coords220[:,0] = seeds[:,0]+1
    coords220[:,1] = seeds[:,1]+1
    coords220[:,2] = seeds[:,2]-1



    coords210 = np.zeros((len(seeds),3))
    coords210[:,0] = seeds[:,0]+1
    coords210[:,1] = seeds[:,1]
    coords210[:,2] = seeds[:,2]-1



    coords200 = np.zeros((len(seeds),3))
    coords200[:,0] = seeds[:,0]+1
    coords200[:,1] = seeds[:,1]-1
    coords200[:,2] = seeds[:,2]-1

#endregion/

#z=1，8
#region*/
    coords101 = np.zeros((len(seeds),3))
    coords101[:,0] = seeds[:,0]
    coords101[:,1] = seeds[:,1]-1
    coords101[:,2] = seeds[:,2]


    coords121 = np.zeros((len(seeds),3))
    coords121[:,0] = seeds[:,0]
    coords121[:,1] = seeds[:,1]+1
    coords121[:,2] = seeds[:,2]

              
    coords011 = np.zeros((len(seeds),3))
    coords011[:,0] = seeds[:,0]-1
    coords011[:,1] = seeds[:,1]
    coords011[:,2] = seeds[:,2]


    coords001 = np.zeros((len(seeds),3))
    coords001[:,0] = seeds[:,0]-1
    coords001[:,1] = seeds[:,1]-1
    coords001[:,2] = seeds[:,2]



    coords021 = np.zeros((len(seeds),3))
    coords021[:,0] = seeds[:,0]-1
    coords021[:,1] = seeds[:,1]+1
    coords021[:,2] = seeds[:,2]

    coords221 = np.zeros((len(seeds),3))
    coords221[:,0] = seeds[:,0]+1
    coords221[:,1] = seeds[:,1]+1
    coords221[:,2] = seeds[:,2]


    coords211 = np.zeros((len(seeds),3))
    coords211[:,0] = seeds[:,0]+1
    coords211[:,1] = seeds[:,1]
    coords211[:,2] = seeds[:,2]


    coords201 = np.zeros((len(seeds),3))
    coords201[:,0] = seeds[:,0]+1
    coords201[:,1] = seeds[:,1]-1
    coords201[:,2] = seeds[:,2]

#endregion*/

#z=2，9
#region*/
    coords112 = np.zeros((len(seeds),3))
    coords112[:,0] = seeds[:,0]
    coords112[:,1] = seeds[:,1]
    coords112[:,2] = seeds[:,2]+1


    coords102 = np.zeros((len(seeds),3))
    coords102[:,0] = seeds[:,0]
    coords102[:,1] = seeds[:,1]-1
    coords102[:,2] = seeds[:,2]+1



    coords122 = np.zeros((len(seeds),3))
    coords122[:,0] = seeds[:,0]
    coords122[:,1] = seeds[:,1]+1
    coords122[:,2] = seeds[:,2]+1



    coords012 = np.zeros((len(seeds),3))
    coords012[:,0] = seeds[:,0]-1
    coords012[:,1] = seeds[:,1]
    coords012[:,2] = seeds[:,2]+1


    coords002 = np.zeros((len(seeds),3))
    coords002[:,0] = seeds[:,0]-1
    coords002[:,1] = seeds[:,1]-1
    coords002[:,2] = seeds[:,2]+1


    coords022 = np.zeros((len(seeds),3))
    coords022[:,0] = seeds[:,0]-1
    coords022[:,1] = seeds[:,1]+1
    coords022[:,2] = seeds[:,2]+1

    coords222 = np.zeros((len(seeds),3))
    coords222[:,0] = seeds[:,0]+1
    coords222[:,1] = seeds[:,1]+1
    coords222[:,2] = seeds[:,2]+1


    coords212 = np.zeros((len(seeds),3))
    coords212[:,0] = seeds[:,0]+1
    coords212[:,1] = seeds[:,1]
    coords212[:,2] = seeds[:,2]+1

    coords202 = np.zeros((len(seeds),3))
    coords202[:,0] = seeds[:,0]+1
    coords202[:,1] = seeds[:,1]-1
    coords202[:,2] = seeds[:,2]+1

    #endregion*/

    all_coords= np.concatenate((coords000,coords001,coords002,coords010,coords011,coords012,coords020,coords021,coords022,
                coords100,coords101,coords102,coords110,seeds,coords112,coords120,coords121,coords122,
                coords200,coords201,coords202,coords210,coords211,coords212,coords220,coords221,coords222),axis=0)#(27*all_pts,3)

    all_data= eval_volume_at_3d_coordinates(dwi_vol, all_coords)#(n,45) 

    a = np.zeros((27,len(seeds),all_data.shape[-1]))
    for i in range(27):
        start=i*int(len(seeds))
        end = (i+1)*int(len(seeds))
        a[i]=all_data[start:end,:]
    a = a.transpose(1,0,2)
    a = a.reshape(len(seeds),3,3,3,all_data.shape[-1])
    return a

def convert2dwi_patch_v3(dwi_vol, seeds):
    """
    333/999 patch整体偏移,每一步都重新生成坐标映射矩阵
    INPUT:  dwi_vol - signal volume 
            strmls  - nib.streamliens.ArraySequence,期望是一个subj所有纤维束,所有点

    """
    #region*/
    #纤维束全部点的坐标，true cube_center
    cube_center = seeds#(n,3)


    orig_cube_center=np.round(cube_center).astype(int)
    orig_cube_center_x=orig_cube_center[:,0]#(n,)
    orig_cube_center_y=orig_cube_center[:,1]#(n,)
    orig_cube_center_z=orig_cube_center[:,2]#(n,)

    #创建一个数组，大小为(90，108，90),每个位置上都是自己的坐标索引
    dwi_coord_map=[]
    h,w,l,ch=dwi_vol.shape
    for idx in np.ndindex(h,w,l):
        dwi_coord_map.append(idx)
    dwi_coord_map=np.array(dwi_coord_map).reshape(h,w,l,3)


    #从坐标索引矩阵中取出坐标cube
    orig_cube=[dwi_coord_map[orig_cube_center_x[i]-1:orig_cube_center_x[i]+2,
                        orig_cube_center_y[i]-1:orig_cube_center_y[i]+2,
                        orig_cube_center_z[i]-1:orig_cube_center_z[i]+2] for i in range(len(orig_cube_center)) ]
    orig_cube=np.array(orig_cube)#(n,9,9,9)

    
    #计算cube中心点与最近栅格点的各方向偏移
    x_b=cube_center[:,0]-orig_cube_center_x#(n,)
    y_b=cube_center[:,1]-orig_cube_center_y
    z_b=cube_center[:,2]-orig_cube_center_z

    #原cube整体偏移
    a=orig_cube[...,0]+ x_b[:,None,None,None]
    b=orig_cube[...,1]+ y_b[:,None,None,None]
    c=orig_cube[...,2]+ z_b[:,None,None,None]
    cube=np.stack((a,b,c))
    cube=cube.transpose(1,2,3,4,0)#(n,9,9,9,3)
    #endregion*/

    all_pts=cube.reshape(-1,3)#(n_,3)
    dwi_data=eval_volume_at_3d_coordinates(dwi_vol, all_pts)#(n_,ch)
    re_cube=dwi_data.reshape(-1,3,3,3,ch)#(n,9,9,9,ch)
    return re_cube    

def convert2dwi_patch_v4(dwi_vol, seeds):
    """
    999patch 只偏移中心体素,其余取栅格点对应的值
    INPUT:  dwi_vol - signal volume 
            strmls  - nib.streamliens.ArraySequence,期望是一个subj所有纤维束,所有点

    """
    #region*/
    #纤维束全部点的坐标，true cube_center
    cube_center = seeds#(n,3)


    orig_cube_center=np.round(cube_center).astype(int)
    orig_cube_center_x=orig_cube_center[:,0]#(n,)
    orig_cube_center_y=orig_cube_center[:,1]#(n,)
    orig_cube_center_z=orig_cube_center[:,2]#(n,)



    #从坐标索引矩阵中取出坐标cube
    dwi_cube=[dwi_vol[orig_cube_center_x[i]-4:orig_cube_center_x[i]+5,
                        orig_cube_center_y[i]-4:orig_cube_center_y[i]+5,
                        orig_cube_center_z[i]-4:orig_cube_center_z[i]+5] for i in range(len(orig_cube_center)) ]
    dwi_cube=np.array(dwi_cube)#(n,9,9,9,ch)
    #endregion*/

    dwi_cube_center=eval_volume_at_3d_coordinates(dwi_vol, cube_center)#(n,ch)
    dwi_cube[:,4,4,4,:]=dwi_cube_center
    return dwi_cube    

def convert2dwi_patch_v5(dwi_vol,dwi_coord_map, seeds):
    """
    333 patch,需传入dwi_coord_map
    """
    #region*/
    #纤维束全部点的坐标，true cube_center
    cube_center = seeds#(n,3)


    orig_cube_center=np.round(cube_center).astype(int)
    orig_cube_center_x=orig_cube_center[:,0]#(n,)
    orig_cube_center_y=orig_cube_center[:,1]#(n,)
    orig_cube_center_z=orig_cube_center[:,2]#(n,)


    #从坐标索引矩阵中取出坐标cube
    orig_cube=[dwi_coord_map[orig_cube_center_x[i]-1:orig_cube_center_x[i]+2,
                        orig_cube_center_y[i]-1:orig_cube_center_y[i]+2,
                        orig_cube_center_z[i]-1:orig_cube_center_z[i]+2] for i in range(len(orig_cube_center)) ]

    
    orig_cube=np.array(orig_cube)#(n,9,9,9)

    
    #计算cube中心点与最近栅格点的各方向偏移
    x_b=cube_center[:,0]-orig_cube_center_x#(n,)
    y_b=cube_center[:,1]-orig_cube_center_y
    z_b=cube_center[:,2]-orig_cube_center_z

    #原cube整体偏移
    a=orig_cube[...,0]+ x_b[:,None,None,None]
    b=orig_cube[...,1]+ y_b[:,None,None,None]
    c=orig_cube[...,2]+ z_b[:,None,None,None]
    cube=np.stack((a,b,c))
    cube=cube.transpose(1,2,3,4,0)#(n,9,9,9,3)
    #endregion*/

    all_pts=cube.reshape(-1,3)#(n_,3)
    dwi_data=eval_volume_at_3d_coordinates(dwi_vol, all_pts)#(n_,ch)
    re_cube=dwi_data.reshape(-1,3,3,3,dwi_vol.shape[-1])#(n,9,9,9,ch)
    return re_cube  

def convert2dwi_patch_v6(dwi_vol,dwi_coord_map, seeds):
    """
    999 patch,需传入dwi_coord_map
    """
    #region*/
    #纤维束全部点的坐标，true cube_center
    cube_center = seeds#(n,3)


    orig_cube_center=np.round(cube_center).astype(int)
    orig_cube_center_x=orig_cube_center[:,0]#(n,)
    orig_cube_center_y=orig_cube_center[:,1]#(n,)
    orig_cube_center_z=orig_cube_center[:,2]#(n,)


    #从坐标索引矩阵中取出坐标cube
    orig_cube=[dwi_coord_map[orig_cube_center_x[i]-4:orig_cube_center_x[i]+5,
                        orig_cube_center_y[i]-4:orig_cube_center_y[i]+5,
                        orig_cube_center_z[i]-4:orig_cube_center_z[i]+5] for i in range(len(orig_cube_center)) ]
    orig_cube=np.array(orig_cube)#(n,9,9,9)

    
    #计算cube中心点与最近栅格点的各方向偏移
    x_b=cube_center[:,0]-orig_cube_center_x#(n,)
    y_b=cube_center[:,1]-orig_cube_center_y
    z_b=cube_center[:,2]-orig_cube_center_z

    #原cube整体偏移
    a=orig_cube[...,0]+ x_b[:,None,None,None]
    b=orig_cube[...,1]+ y_b[:,None,None,None]
    c=orig_cube[...,2]+ z_b[:,None,None,None]
    cube=np.stack((a,b,c))
    cube=cube.transpose(1,2,3,4,0)#(n,9,9,9,3)
    #endregion*/

    all_pts=cube.reshape(-1,3)#(n_,3)
    dwi_data=eval_volume_at_3d_coordinates(dwi_vol, all_pts)#(n_,ch)
    re_cube=dwi_data.reshape(-1,9,9,9,dwi_vol.shape[-1])#(n,9,9,9,ch)
    return re_cube

def convert2dwi_patch_v7(dwi_vol,dwi_coord_map,dwi2struc,std2struc, seeds):
    """
    with coords as one feature

    """
    #纤维束全部点的坐标，true cube_center
    cube_center = seeds#(n,3)
    seedstruc=transform(cube_center,dwi2struc)
    if np.isnan(seedstruc).any()==True:
        print('2',np.isnan(seedstruc).any())
    seedmni=std2struc.transform(seedstruc,'voxel','voxel')
    seedmni=np.nan_to_num(seedmni)
    if np.isnan(seedmni).any()==True:
        print('3',np.isnan(seedmni).any())
    # seedmni[:,0]=seedmni[:,0]/91
    # seedmni[:,1]=seedmni[:,1]/109
    # seedmni[:,2]=seedmni[:,2]/91
    seedmni=np.floor(seedmni/3)
    if np.isnan(seedmni).any()==True:
        print('4',np.isnan(seedmni).any())


    orig_cube_center=np.round(cube_center).astype(int)
    orig_cube_center_x=orig_cube_center[:,0]#(n,)
    orig_cube_center_y=orig_cube_center[:,1]#(n,)
    orig_cube_center_z=orig_cube_center[:,2]#(n,)


    #从坐标索引矩阵中取出坐标cube
    orig_cube=[dwi_coord_map[orig_cube_center_x[i]-1:orig_cube_center_x[i]+2,
                        orig_cube_center_y[i]-1:orig_cube_center_y[i]+2,
                        orig_cube_center_z[i]-1:orig_cube_center_z[i]+2] for i in range(len(orig_cube_center)) ]
    orig_cube=np.array(orig_cube)#(n,9,9,9)

    
    #计算cube中心点与最近栅格点的各方向偏移
    x_b=cube_center[:,0]-orig_cube_center_x#(n,)
    y_b=cube_center[:,1]-orig_cube_center_y
    z_b=cube_center[:,2]-orig_cube_center_z

    #原cube整体偏移
    a=orig_cube[...,0]+ x_b[:,None,None,None]
    b=orig_cube[...,1]+ y_b[:,None,None,None]
    c=orig_cube[...,2]+ z_b[:,None,None,None]
    cube=np.stack((a,b,c))
    cube=cube.transpose(1,2,3,4,0)#(n,9,9,9,3)

    all_pts=cube.reshape(-1,3)#(n_,3)
    dwi_data=eval_volume_at_3d_coordinates(dwi_vol, all_pts)#(n_,ch)
    re_cube=dwi_data.reshape(-1,3,3,3,dwi_vol.shape[-1])#(n,9,9,9,ch)
    return re_cube,seedmni

def convert2dwi_patch_v8(dwi_vol,offsets_matrix, coords):
    batch_size=len(coords)

    # 创建一个布尔掩码，标记为 (0, 0, 0) 的体素
    mask = np.all(coords == 0, axis=1)  # 形状为 (batch_size,)

    coords_repeated = np.repeat(coords, offsets_matrix.shape[0], axis=0)  # 形状为 (batch_size * 27, 3)
    # 计算邻域坐标
    neighborhood = coords_repeated + np.tile(offsets_matrix, (batch_size, 1))  # 形状为 (batch_size * 27, 3)
    # 需要扩展 mask 到与 coords_repeated 对应的结构
    expanded_mask = np.repeat(mask, offsets_matrix.shape[0])  # 形状为 (batch_size * 27,)

    # 使用扩展的 mask 设置邻域
    neighborhood[expanded_mask] = 0  # 广播赋值，形状为 (batch_size * 27, 3)

    dwi_neibor=eval_volume_at_3d_coordinates(dwi_vol, neighborhood)#(batch_size * 27,ch)

    # 重塑为 (batch_size, 3, 3, 3, 3)，最后一维存储三维坐标
    dwi_neibor = dwi_neibor.reshape(batch_size, 3, 3, 3, dwi_vol.shape[-1])
    return dwi_neibor
