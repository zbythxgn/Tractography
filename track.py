import os
import sys
import argparse
import pickle
from contextlib import nullcontext
import torch
import torch.nn.functional as F

import numpy as np
from os.path import join as pjoin
import time
import dipy
import nibabel as nib
from nibabel.streamlines import Tractogram
from test_utils import Timer
from VoxCoordDataLoader import VoxCoordDataLoader
from test_utils import idx2direction, map_coordinates_3d_4d, convert2dwi_patch_v8
from train_utils import eval_volume_at_3d_coordinates

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
from model import GPTConfig, GPT


max_new_tokens = 201 # number of tokens generated in each sample
seed = 1337
device = 'cuda:0' # default device, will be overridden by --device parameter
dtype = 'float32' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Stopping Constant
STOPPING_MASK =       int('00000001', 2)
STOPPING_LENGTH =     int('00000010', 2)
STOPPING_CURVATURE =  int('00000100', 2)
STOPPING_LIKELIHOOD = int('00001000', 2)

# neighbourhood_matrix
axes = np.array([-1, 0, 1])
offsets_matrix = np.array(np.meshgrid(axes, axes, axes, indexing='ij')).reshape(3, -1).T


def sample_dwi_patches_gpu(dwi_volume, offsets, coords):
    """Sample 3x3x3 DWI patches on the device with trilinear interpolation."""
    batch_size = coords.shape[0]
    neighborhood = coords[:, None, :] + offsets[None, :, :]
    neighborhood = torch.where(
        (coords == 0).all(dim=1, keepdim=True).unsqueeze(-1),
        torch.zeros_like(neighborhood),
        neighborhood,
    )

    x_size = dwi_volume.shape[-1]
    y_size = dwi_volume.shape[-2]
    z_size = dwi_volume.shape[-3]
    scale = neighborhood.new_tensor([x_size - 1, y_size - 1, z_size - 1])
    grid = 2 * neighborhood / scale - 1
    grid = grid.view(1, batch_size, offsets.shape[0], 1, 3)
    sampled = F.grid_sample(
        dwi_volume, grid, mode='bilinear', padding_mode='border', align_corners=True,
    )
    return sampled[0, :, :, :, 0].permute(1, 2, 0).reshape(batch_size, 3, 3, 3, -1)



# 设置追踪时间全局记录变量
model_start_event = torch.cuda.Event(enable_timing=True)
model_end_event = torch.cuda.Event(enable_timing=True)




def is_flag_set(flags, ref_flag):
    return ((flags.astype(np.uint8) & ref_flag) >> np.log2(ref_flag).astype(np.uint8)).astype(bool)


def count_flags(flags, ref_flag):
    return is_flag_set(flags, ref_flag).sum()


def make_is_outside_mask(mask, affine, threshold=0, displacement=0):
    def _is_outside_mask(streamlines, timestep, *args):
        last_coordinates = streamlines[:, timestep, :]
        mask_values = map_coordinates_3d_4d(mask, last_coordinates, affine=affine, order=1, displacement=displacement)
        return mask_values < threshold

    return _is_outside_mask

def make_is_too_long(max_length):
    def _is_too_long(streamlines, timestep, *args):
        return np.asarray([timestep + 1 > max_length] * len(streamlines))

    return _is_too_long

def make_is_too_curvy(max_theta):
    max_theta = np.deg2rad(max_theta)

    def _is_too_curvy(streamlines, timestep, *args):
        if timestep + 1 < 3:
            return np.asarray([False] * len(streamlines))

        last_segments = streamlines[:, timestep] - streamlines[:, timestep - 1]
        before_last_segments = streamlines[:, timestep - 1] - streamlines[:, timestep - 2]
        zero_mask_last = np.all(last_segments == 0, axis=1)
        zero_mask_before_last = np.all(before_last_segments == 0, axis=1)

        last_segments[~zero_mask_last] /= np.sqrt(np.sum(last_segments[~zero_mask_last] ** 2, axis=1, keepdims=True))
        before_last_segments[~zero_mask_before_last] /= np.sqrt(np.sum(before_last_segments[~zero_mask_before_last] ** 2, axis=1, keepdims=True))

        last_segments[zero_mask_last] = 0
        before_last_segments[zero_mask_before_last] = 0

        cos = np.sum(last_segments * before_last_segments, axis=1)
        eps = 1e-6
        cos = [1.0 if 1.0 < i < 1.0 + eps else i for i in cos]
        cos = [-1.0 if -1.0 - eps < i < -1.0 else i for i in cos]

        angles = np.arccos(cos)
        return angles > max_theta

    return _is_too_curvy


def make_is_stopping(stopping_criteria):
    def _is_stopping(streamlines, timestep, to_check=None):
        if to_check is None:
            idx = np.arange(len(streamlines))
        elif isinstance(to_check, np.ndarray) and to_check.dtype == np.bool:
            assert len(to_check) == len(streamlines)
            idx = np.where(to_check)[0]
        else:
            idx = to_check

        undone = np.ones(len(idx), dtype=bool)
        flags = np.zeros(len(idx), dtype=np.uint8)
        for flag, stopping_criterion in stopping_criteria.items():
            done = stopping_criterion(streamlines[idx], timestep)
            undone[done] = False
            flags[done] |= flag

        done = np.logical_not(undone)
        return idx[np.where(undone)[0]], idx[np.where(done)[0]], flags[done]

    return _is_stopping


class Tracker2(object):
    def __init__(self, model, is_stopping, dwi, dwi_coord_map, flip_x=False, flip_y=False, flip_z=False, device='cpu', block_size=96):
        self.model = model
        self.device = device
        self.dwi = dwi
        self.dwi_coord_map = dwi_coord_map
        self._is_stopping = is_stopping
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z
        self.block_size = block_size
        self.done = None
        self.done_new = None
        self._is_cuda = torch.device(device).type == 'cuda'
        self.dwi_device = torch.as_tensor(dwi, dtype=torch.float32, device=device)
        self.dwi_device = self.dwi_device.permute(3, 2, 1, 0).unsqueeze(0).contiguous()
        self.offsets_device = torch.as_tensor(offsets_matrix, dtype=torch.float32, device=device)

    def _sample_dwi_patches(self, coords):
        if self._is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            coords_device = torch.as_tensor(coords, dtype=torch.float32, device=self.device)
            patches = sample_dwi_patches_gpu(self.dwi_device, self.offsets_device, coords_device)
            end_event.record()
            end_event.synchronize()
            return patches, start_event.elapsed_time(end_event) / 1000

        start_time = time.perf_counter()
        coords_device = torch.as_tensor(coords, dtype=torch.float32, device=self.device)
        patches = sample_dwi_patches_gpu(self.dwi_device, self.offsets_device, coords_device)
        return patches, time.perf_counter() - start_time

    def is_stopping(self, sprouts, timestep):
        undone, done, stopping_flags = self._is_stopping(sprouts, timestep)
        return undone, done, stopping_flags

    def is_ripe(self):
        return len(self.undone) == 0

    def plant(self, seeds, sprouts, sprouts_dwi, sprouts_dwi_t, t_prev=1):
        if seeds.ndim == 2:
            seeds = seeds[:, None, :]

        if t_prev == seeds.shape[1]:
            sprouts[:len(seeds), :t_prev, :] = seeds.copy()
            self.sprouts_stop = np.ones((len(seeds), 1))
            sprouts_dwi_tmp, _ = self._sample_dwi_patches(sprouts[:len(seeds), t_prev - 1, :])
            sprouts_dwi_t[:len(seeds), t_prev - 1].copy_(sprouts_dwi_tmp)
            self.all_id = self.undone = np.arange(len(seeds))
            self.done = None
            self.done_new = None
        else:
            print('error')

    def _grow_step(self, sprouts, sprouts_dwi, sprouts_dwi_t, step_size, save_weights, timestep):
        with torch.no_grad():
            with ctx:
                model_start_event.record()
                outputs = self.model.generate(idx=sprouts_dwi_t[self.undone, max((0, timestep - self.block_size)):timestep, :, :, :, :])
                model_end_event.record()
                model_end_event.synchronize()

                outputs = outputs.cpu().numpy()


        directions = outputs / np.sqrt(np.sum(outputs ** 2, axis=-1, keepdims=True) + 1e-6)


        if self.flip_x:
            directions[:, 0] *= -1
        if self.flip_y:
            directions[:, 1] *= -1
        if self.flip_z:
            directions[:, 2] *= -1

        if step_size is not None:
            normalized_directions = directions / np.sqrt(np.sum(directions ** 2, axis=1, keepdims=True))
            directions = normalized_directions * step_size

        new_coord = sprouts[self.undone, timestep - 1, :] + directions
        sprouts[self.undone, timestep, :] = new_coord

        new_dwi, sample_duration = self._sample_dwi_patches(new_coord)

        undone_device = torch.as_tensor(self.undone, dtype=torch.long, device=self.device)
        sprouts_dwi_t[:, timestep].index_copy_(0, undone_device, new_dwi)

        return sprouts, sprouts_dwi, sprouts_dwi_t

    def grow(self,  step_size, save_weights, sprouts, sprouts_dwi, sprouts_dwi_t, timestep):
        sprouts, sprouts_dwi, sprouts_dwi_t = self._grow_step(sprouts, sprouts_dwi, sprouts_dwi_t, step_size, save_weights, timestep)

    def harvest(self, sprouts, sprouts_dwi, sprouts_dwi_t, timestep):
        undone_t, done_t, stopping_flags_t = self.is_stopping(sprouts, timestep)
        if self.done is None:
            self.done = done_t
            self.done_new = done_t
        else:
            self.done_new = np.asarray(list(set(done_t) - set(self.done))).astype(int)
            self.done = np.asarray(list(set(done_t) | set(self.done)))
            self.undone = np.asarray(list(set(self.all_id) - set(self.done))).astype(int)

        streamlines = list(sprouts[self.done_new, :timestep])
        tractogram = Tractogram(streamlines=streamlines,
                                data_per_streamline={"stopping_flags": stopping_flags_t[np.isin(done_t, self.done_new)]})
        return tractogram


class BackwardTracker2(Tracker2):
    """优化后的 GPU 批处理反向追踪器"""
    def plant(self, seeds, sprouts, sprouts_dwi, sprouts_dwi_t, t_prev=1):
        self.seeds = seeds
        self.nb_init_steps = np.asarray([len(s) for s in seeds])
        self.all_id = self.undone = np.arange(len(seeds))
        self.done = None
        self.done_new = None

        if len(seeds) == 0:
            return

        tmp = np.asarray([s[0] for s in seeds])
        sprouts[:len(seeds), 0, :] = tmp
        self.sprouts_stop = np.ones((len(seeds), 1))
        
        sprouts_dwi_tmp, _ = self._sample_dwi_patches(tmp)
        sprouts_dwi_t[:len(seeds), 0].copy_(sprouts_dwi_tmp)

    def is_stopping(self, sprouts, timestep):
        undone, done, stopping_flags = self._is_stopping(sprouts, timestep)

        if len(done) > 0:
            # 在历史轨迹重现阶段（timestep + 1 <= nb_init_steps），禁止将纤维束标记为终止
            init_undone = (self.nb_init_steps >= timestep + 1)
            init_undone_done = [idx for idx in done if init_undone[idx]]
            
            if len(init_undone_done) > 0:
                undone = np.r_[undone, init_undone_done].astype(int)
                truly_done_mask = np.logical_not(init_undone[done])
                done = done[truly_done_mask]
                stopping_flags = stopping_flags[truly_done_mask]

        return undone, done, stopping_flags

    def _grow_step(self, sprouts, sprouts_dwi, sprouts_dwi_t,  step_size, save_weights, timestep):
        with torch.no_grad():
            with ctx:
                model_start_event.record()
                outputs = self.model.generate(idx=sprouts_dwi_t[self.undone, max((0, timestep - self.block_size)):timestep, :, :, :, :])
                model_end_event.record()
                model_end_event.synchronize()

                outputs = outputs.cpu().numpy()

        directions = outputs / np.sqrt(np.sum(outputs ** 2, axis=-1, keepdims=True) + 1e-6)


        if self.flip_x:
            directions[:, 0] *= -1
        if self.flip_y:
            directions[:, 1] *= -1
        if self.flip_z:
            directions[:, 2] *= -1

        if step_size is not None:
            normalized_directions = directions / np.sqrt(np.sum(directions ** 2, axis=1, keepdims=True))
            directions = normalized_directions * step_size

        new_coord = sprouts[self.undone, timestep - 1, :] + directions

        # 向量化判断：初始化阶段强制覆盖坐标为历史反向轨迹点
        init_mask = (timestep < self.nb_init_steps[self.undone])
        if np.any(init_mask):
            init_indices_in_undone = np.where(init_mask)[0]
            init_orig_indices = self.undone[init_mask]
            new_coord[init_indices_in_undone] = np.array([self.seeds[i][timestep] for i in init_orig_indices])

        sprouts[self.undone, timestep, :] = new_coord

        # 统一在 GPU 上采样 DWI 邻域特征
        new_dwi, sample_duration = self._sample_dwi_patches(new_coord)

        undone_device = torch.as_tensor(self.undone, dtype=torch.long, device=self.device)
        sprouts_dwi_t[:, timestep].index_copy_(0, undone_device, new_dwi)

        return sprouts, sprouts_dwi, sprouts_dwi_t

def track( tracker, seeds, step_size, is_stopping, sprouts_dwi, sprouts, sprouts_dwi_t, nb_retry=0, nb_backtrack_steps=0, verbose=False, save_weights=False, t_prev=1):
    tractogram = None
    timestep = 1
    tracker.plant(seeds, sprouts, sprouts_dwi, sprouts_dwi_t, t_prev)

    while not tracker.is_ripe():
        if verbose:
            print("pts: {}/{} ({:,} remaining)".format(timestep + 1, is_stopping.max_nb_points, len(sprouts)), end="")

        tracker.grow( step_size, save_weights, sprouts, sprouts_dwi, sprouts_dwi_t, timestep)

        if tractogram is None:
            tractogram = tracker.harvest(sprouts, sprouts_dwi, sprouts_dwi_t, timestep)
        else:
            tractogram += tracker.harvest(sprouts, sprouts_dwi, sprouts_dwi_t, timestep)


        if verbose and nb_retry == 0:
            print("")

        timestep += 1

    return tractogram

def batch_track( model, dwi, seeds, dwi_coord_map, flip_x, flip_y, flip_z, verbose, step_size, batch_size, is_stopping, block_size, max_nb_points):
    if batch_size is None:
        batch_size = len(seeds)

    nb_retry = 0
    nb_backtrack_steps = 0

    TrackerCls = Tracker2
    BackwardTrackerCls = BackwardTracker2

    sprouts = np.zeros((batch_size, max_nb_points, 3))

    sprouts_dwi = None
    sprouts_dwi_t = torch.zeros(
        (batch_size, max_nb_points, 3, 3, 3, dwi.shape[-1]),
        dtype=torch.float32, device=device,
    )

    while True:
        try:
            time.sleep(1)
            print("Trying to track {:,} streamlines at the same time.".format(batch_size))
            tractogram = None
            batch_nb = 0
            save_weights = False

            for start in range(0, len(seeds), batch_size):
                print("{:,} / {:,}".format(start, len(seeds)))
                end = start + batch_size

                # 1. Forward tracking
                tracker = TrackerCls(model, is_stopping, dwi, dwi_coord_map,
                                     flip_x, flip_y, flip_z, device, block_size)
                batch_tractogram = track( tracker=tracker, seeds=seeds[start:end], step_size=step_size, is_stopping=is_stopping,
                                         sprouts=sprouts, sprouts_dwi=sprouts_dwi, sprouts_dwi_t=sprouts_dwi_t,
                                         nb_retry=nb_retry, nb_backtrack_steps=nb_backtrack_steps, verbose=verbose, save_weights=save_weights, t_prev=1)

                stopping_flags = batch_tractogram.data_per_streamline['stopping_flags'].astype(np.uint8)
                print("Forward pass stopped because of - mask: {:,}\t curv: {:,}\t length: {:,}\t likelihood: {:,}".format(
                    count_flags(stopping_flags, STOPPING_MASK),
                    count_flags(stopping_flags, STOPPING_CURVATURE),
                    count_flags(stopping_flags, STOPPING_LENGTH),
                    count_flags(stopping_flags, STOPPING_LIKELIHOOD)))

                # 2. Backward tracking (在正向结果上翻转继续反向追踪)
                save_weights = False
                tracker = BackwardTrackerCls(model, is_stopping, dwi, dwi_coord_map,
                                            flip_x, flip_y, flip_z, device, block_size)
                streamlines = [s[::-1] for s in batch_tractogram.streamlines]  # 翻转前半段正向纤维束
                t_prev = len(streamlines)

                batch_tractogram = track( tracker=tracker, seeds=streamlines, step_size=step_size, is_stopping=is_stopping,
                                         sprouts=sprouts, sprouts_dwi=sprouts_dwi, sprouts_dwi_t=sprouts_dwi_t,
                                         nb_retry=nb_retry, nb_backtrack_steps=nb_backtrack_steps, verbose=verbose, save_weights=save_weights, t_prev=t_prev)

                stopping_flags = batch_tractogram.data_per_streamline['stopping_flags'].astype(np.uint8)
                print("Backward pass stopped because of - mask: {:,}\t curv: {:,}\t length: {:,}\t likelihood: {:,}".format(
                    count_flags(stopping_flags, STOPPING_MASK),
                    count_flags(stopping_flags, STOPPING_CURVATURE),
                    count_flags(stopping_flags, STOPPING_LENGTH),
                    count_flags(stopping_flags, STOPPING_LIKELIHOOD)))

                if tractogram is None:
                    tractogram = batch_tractogram
                else:
                    tractogram += batch_tractogram

                batch_nb += 1

            return tractogram

        except MemoryError:
            print("{:,} streamlines is too much!".format(batch_size))
            batch_size //= 2
            if batch_size < 0:
                raise MemoryError("Might needs a bigger graphic card!")

        except RuntimeError as e:
            if "out of memory" in e.args[0]:
                print("{:,} streamlines is too much!".format(batch_size))
                batch_size //= 2
                if batch_size < 0:
                    raise MemoryError("Might needs a bigger graphic card!")
            else:
                raise e

def get_max_angle_from_curvature(curvature, step_size):
    theta = 2. * np.arcsin(step_size / (2. * curvature))
    if np.isnan(theta) or theta > np.pi / 2 or theta <= 0:
        theta = np.pi / 2.0
    return theta


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(args_list=None):
    """
    解析命令行参数，支持从 sys.argv 读取或直接传入参数列表 list
    """
    parser = argparse.ArgumentParser(description="Tractography Tracking Script")
    parser.add_argument('--device', type=str, default="cuda:0", help="GPU or CPU device (e.g. cuda:0, cuda:1)")
    parser.add_argument('--sub', type=str, default="sub-01", help="Subject ID")
    parser.add_argument('--dwi_path', type=str, required=True, help="Path to DWI image")
    parser.add_argument('--bvec', '--b_vec', dest='bvec', type=str, required=True, help="Path to bvec file")
    parser.add_argument('--bval', '--b_val', dest='bval', type=str, required=True, help="Path to bval file")
    parser.add_argument('--tracking_mask', type=str, required=True, help="Path to WM tracking mask")
    parser.add_argument('--seeding_mask', type=str, required=True, help="Path to WM seeding mask")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Model checkpoint path")
    parser.add_argument('--seeds_per_vox', '--nb_seeds_per_voxel', dest='seeds_per_vox', type=int, default=1, help="Seeds per voxel")
    parser.add_argument('--out_dir', type=str, default="/path/to/save/", help="Output directory")
    parser.add_argument('--out_tck', '--postfix', dest='out_tck', type=str, default="_tracked.tck", help="Output tck filename or postfix")
    parser.add_argument('--batchsize', '--batch_size', dest='batchsize', type=int, default=1024, help="Tracking batch size")
    parser.add_argument('--use_seeding_mask', type=str2bool, default=False, help="Flag for using explicit seeding mask")

    args = parser.parse_args(args_list)

    # 1. 组装数据加载参数
    paramss = {
        'dwi_path': args.dwi_path,
        'b_vec': args.bvec,
        'b_val': args.bval,
        'mode': 'track',
        'training_type': 'regression',
        'tracking_mask': args.tracking_mask,
        'wm_mask_path': args.tracking_mask,
        'seeding_mask': args.seeding_mask,
        'coords_map': False
    }

    # 2. 组装追踪参数
    t_params = {
        'device': args.device,
        'ckpt_path': args.ckpt_path,
        'out_dir': args.out_dir,
        'nb_seeds_per_voxel': args.seeds_per_vox,
        'postfix': args.out_tck,
        'step_size': 1,
        'min_length': 20,
        'max_length': 202,
        'theta': 20,
        'curvature': None,
        'discard_stopped_by_curvature': False,
        'mask_threshold': 0.05,
        'batch_size': args.batchsize,
        'flip_x': False, 'flip_y': False, 'flip_z': False,
        'save_rejected': False,
        'verbose': False,
        'dilate_mask': False,
        'seeding_rng_seed': 1234,
        'use_seeding_mask': args.use_seeding_mask
    }

    return paramss, t_params


def main(paramss, t_params):
    """
    核心追踪函数接口，接受配置字典 paramss 和 t_params
    """
    global device, device_type, ctx
    device = t_params['device']
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ckpt_path = t_params['ckpt_path']
    out_dir = t_params['out_dir']
    postfix = t_params['postfix']
    step_size = t_params['step_size']
    min_length = t_params['min_length']
    max_length = t_params['max_length']
    theta = t_params['theta']
    curvature = t_params['curvature']
    discard_stopped_by_curvature = t_params['discard_stopped_by_curvature']
    mask_threshold = t_params['mask_threshold']
    batch_size = t_params['batch_size']
    flip_x = flip_y = flip_z = t_params['flip_x']
    save_rejected = t_params['save_rejected']
    verbose = t_params['verbose']
    dilate_mask = t_params['dilate_mask']
    seeding_rng_seed = t_params['seeding_rng_seed']
    nb_seeds_per_voxel = t_params['nb_seeds_per_voxel']
    use_seeding_mask = t_params['use_seeding_mask']

    with Timer("Loading DWIs"):
        origdata = VoxCoordDataLoader(paramss)
        origdata.dwi_sh_coeff = origdata.get_spherical_harmonics_coefficients(origdata.dwi, origdata.bvals,
                                                                              origdata.bvecs, subname='other', sh_order=6,
                                                                              mean_centering=True)
        l, w, h, c = origdata.dwi_sh_coeff.shape
        dwi_padz = np.zeros((l + 12, w + 12, h + 12, c))
        dwi_padz[6:-6, 6:-6, 6:-6, :] = origdata.dwi_sh_coeff
        weights = dwi_padz
        affine_rasmm2dwivox = np.linalg.inv(origdata.affine)

    with Timer("Loading model"):
        if init_from == 'resume':
            checkpoint = torch.load(ckpt_path, map_location=device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                    if k.endswith('temperature'):
                        del state_dict[k[len(unwanted_prefix):]]
            model.load_state_dict(state_dict)
        elif init_from.startswith('gpt2'):
            model = GPT.from_pretrained(init_from, dict(dropout=0.0))

        print("blocksize of model is:{}".format(gptconf.block_size))
        model.eval()
        model.to(device)
        if compile:
            model = torch.compile(model)

    mask = None
    with Timer("Loading mask"):
        mask_nii = nib.load(paramss['wm_mask_path'])
        mask_o = mask_nii.get_fdata()
        mask = mask_o
        disp = 6

        affine_maskvox2dwivox = np.dot(affine_rasmm2dwivox, mask_nii.affine)
        if dilate_mask:
            import scipy
            mask = scipy.ndimage.morphology.binary_dilation(mask).astype(mask.dtype)

    with Timer("Generating seeds"):
        wm_mask_nii = nib.load(paramss['seeding_mask'])
        wm_mask_o = wm_mask_nii.get_fdata().astype("float32")
        wm_mask = wm_mask_o
        affine_seedsvox2dwivox = np.dot(affine_rasmm2dwivox, wm_mask_nii.affine)
        rng = np.random.RandomState(seeding_rng_seed)

        seeds = []
        if not use_seeding_mask:
            mask_idxs = np.array(np.where(wm_mask)).T
            for idx in mask_idxs:
                seeds_in_voxel = idx + rng.uniform(-0.5, 0.5, size=(nb_seeds_per_voxel, 3))
                seeds_in_voxel = nib.affines.apply_affine(affine_seedsvox2dwivox, seeds_in_voxel)
                seeds.extend(seeds_in_voxel)
        else:
            for i in range(10):
                tmp = i + 1
                mask_idxs1 = np.array(np.where(wm_mask == tmp)).T
                for idx in mask_idxs1:
                    seeds_in_voxel = idx + rng.uniform(-0.5, 0.5, size=(tmp, 3))
                    seeds_in_voxel = nib.affines.apply_affine(affine_seedsvox2dwivox, seeds_in_voxel)
                    seeds.extend(seeds_in_voxel)
        seeds = np.array(seeds, dtype=np.float32)
        seeds = seeds + 6


    with Timer("Generate coords map"):
        dwi_coord_map = []

    with Timer("Tracking in the diffusion voxel space"):
        voxel_sizes = np.asarray(origdata.voxel_sizes)
        if not np.all(voxel_sizes == origdata.voxel_sizes[0]):
            print("* Careful voxel are anisotropic {}!".format(tuple(voxel_sizes)))

        max_nb_points = int(np.ceil(max_length / step_size))
        step_size = np.float32(step_size / voxel_sizes.max())

        if theta is not None:
            theta = np.deg2rad(theta)
        elif curvature is not None and curvature > 0:
            theta = get_max_angle_from_curvature(curvature, step_size)
        else:
            theta = np.deg2rad(45)

        print("Angle: {}".format(np.rad2deg(theta)))
        print("Step size (vox): {}".format(step_size))
        print("Max nb. points: {}".format(max_nb_points))

        is_outside_mask = make_is_outside_mask(mask, affine_maskvox2dwivox, threshold=mask_threshold, displacement=disp)
        is_too_long = make_is_too_long(max_nb_points)
        is_too_curvy = make_is_too_curvy(np.rad2deg(theta))
        is_stopping = make_is_stopping({STOPPING_MASK: is_outside_mask,
                                        STOPPING_LENGTH: is_too_long,
                                        STOPPING_CURVATURE: is_too_curvy})
        is_stopping.max_nb_points = max_nb_points

        tractogram = batch_track( model, weights, seeds, dwi_coord_map,
                                 flip_x, flip_y, flip_z,
                                 verbose,
                                 step_size=step_size,
                                 is_stopping=is_stopping,
                                 batch_size=batch_size,
                                 block_size=gptconf.block_size,
                                 max_nb_points=max_nb_points)

        tractogram.streamlines._data = tractogram.streamlines._data - 6
        tractogram.affine_to_rasmm = origdata.affine
        tractogram.to_world()

    nb_streamlines = len(tractogram)

    if save_rejected:
        rejected_tractogram = Tractogram()
        rejected_tractogram.affine_to_rasmm = tractogram._affine_to_rasmm

    print("Generated {:,} (compressed) streamlines".format(nb_streamlines))
    with Timer("Cleaning streamlines", newline=True):
        if save_rejected:
            rejected_tractogram += tractogram[np.array(list(map(len, tractogram))) <= 0]

        tractogram = tractogram[np.array(list(map(len, tractogram))) > 0]
        print("Removed {:,} empty streamlines".format(nb_streamlines - len(tractogram)))

        nb_streamlines = len(tractogram)
        lengths = dipy.tracking.streamline.length(tractogram.streamlines)

        if save_rejected:
            rejected_tractogram += tractogram[lengths < min_length]

        tractogram = tractogram[lengths >= min_length]
        lengths = lengths[lengths >= min_length]
        if len(lengths) > 0:
            print("Average length: {:.2f} mm.".format(lengths.mean()))
            print("Minimum length: {:.2f} mm. Maximum length: {:.2f}".format(lengths.min(), lengths.max()))
        print("Removed {:,} streamlines smaller than {:.2f} mm".format(nb_streamlines - len(tractogram), min_length))
        
        if discard_stopped_by_curvature:
            nb_streamlines = len(tractogram)
            stopping_curvature_flag_is_set = is_flag_set(tractogram.data_per_streamline['stopping_flags'][:, 0], STOPPING_CURVATURE)

            if save_rejected:
                rejected_tractogram += tractogram[stopping_curvature_flag_is_set]

            tractogram = tractogram[np.logical_not(stopping_curvature_flag_is_set)]
            print("Removed {:,} streamlines stopped for having a curvature higher than {:.2f} degree".format(
                nb_streamlines - len(tractogram), np.rad2deg(theta)))

    with Timer("Saving {:,} (compressed) streamlines".format(len(tractogram))):
        if os.path.isabs(postfix):
            save_path = postfix
        else:
            save_path = pjoin(out_dir, postfix)

        try:
            os.makedirs(os.path.dirname(save_path))
        except:
            pass

        print("Saving to {}".format(save_path))
        nib.streamlines.save(tractogram, save_path)

    if save_rejected:
        with Timer("Saving {:,} (compressed) rejected streamlines".format(len(rejected_tractogram))):
            rejected_save_path = save_path.replace(".tck", "_rejected.tck")
            try:
                os.makedirs(os.path.dirname(rejected_save_path))
            except:
                pass

            print("Saving rejected streamlines to {}".format(rejected_save_path))
            nib.streamlines.save(rejected_tractogram, rejected_save_path)


if __name__ == "__main__":
    # 命令行直接运行入口
    paramss, t_params = parse_args()
    main(paramss, t_params)