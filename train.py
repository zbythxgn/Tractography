import os
import sys
import time
import math
import argparse
import bisect
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils import data

from VoxCoordDataLoader import VoxCoordDataLoader
from train_utils import convert2dwi_patch
from model import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Tractography GPT Training")
    
    # 基础路径参数
    parser.add_argument('--data_root', type=str, default='/path/to/tractoinferno', help='Root path to dataset')
    parser.add_argument('--out_dir', type=str, default='/path/to/save/', help='Directory to save checkpoints')
    parser.add_argument('--balance_file', type=str, default='None', help='Path to class weight dict (.npy) or None')
    
    # 受试者 ID 列表参数
    parser.add_argument('--train_subj', nargs='+', type=str, 
                        default=["1018","1042","1063","1087","1148","1203","1229","1240","1282","1025",
                                 "1106","1122","1160","1181","1167","1073","1001","1035","1091","1214"], 
                        help='List of training subject IDs')
    parser.add_argument('--valid_subj', nargs='+', type=str, 
                        default=["1000","1097","1247","1053"], 
                        help='List of validation subject IDs')
    parser.add_argument('--test_subj', nargs='+', type=str, 
                        default=["1006"], 
                        help='List of testing subject IDs')

    # 各数据集划分对应的相对子路径参数
    parser.add_argument('--train_subpath', type=str, default='trainset', help='Relative subpath for training set')
    parser.add_argument('--valid_subpath', type=str, default='validset', help='Relative subpath for validation set')
    parser.add_argument('--test_subpath', type=str, default='testset', help='Relative subpath for test set')
    
    # 文件名与文件格式模板参数 (使用 {sub} 作为受试者编号占位符)
    parser.add_argument('--dwi_template', type=str, default='sub-{sub}__dwi.nii.gz', help='DWI filename template')
    parser.add_argument('--wm_mask_template', type=str, default='sub-{sub}__mask_wm.nii.gz', help='WM mask filename template')
    parser.add_argument('--bval_template', type=str, default='sub-{sub}__dwi.bval', help='bval filename template')
    parser.add_argument('--bvec_template', type=str, default='sub-{sub}__dwi.bvec', help='bvec filename template')
    parser.add_argument('--tractogram_folder', type=str, default='1mm', help='Subfolder containing tractogram files')

    # 训练配置参数
    parser.add_argument('--device', type=str, default='cuda:0', help='Target execution device')
    parser.add_argument('--ckpt_sn', type=str, default='ckpt.pt', help='Checkpoint output filename')
    parser.add_argument('--wandb_run_name', type=str, default='run_01', help='Wandb logging run name')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per device')
    parser.add_argument('--num_epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--block_size', type=int, default=96, help='Sequence block size')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=16, help='DataLoader workers')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Step interval for evaluation')
    return parser.parse_args()

def load_subject_data(subj_list, subpath, data_root, fn_templates):
    """统一的受试者数据预处理与加载函数，支持自定义文件名格式与模板"""
    dwi_list, coord_dwi_list, st_list, stbn_list, cum_lens = [], [], [], [], []
    total_samples = 0

    for sub in subj_list:
        sub_path = os.path.join(data_root, subpath, sub)

        # 根据传入的模板格式化生成具体文件名
        dwi_fn = fn_templates['dwi'].format(sub=sub)
        wm_mask_fn = fn_templates['wm_mask'].format(sub=sub)
        bval_fn = fn_templates['bval'].format(sub=sub)
        bvec_fn = fn_templates['bvec'].format(sub=sub)
        tract_folder = fn_templates['tractogram_folder']

        params_exp = {
            'subname': sub,
            'dwi_path': os.path.join(sub_path, dwi_fn),
            'mode': 'train',
            'training_type': 'regression',
            'wm_mask_path': os.path.join(sub_path, wm_mask_fn),
            'b_val': os.path.join(sub_path, bval_fn),
            'b_vec': os.path.join(sub_path, bvec_fn),
            'coords_map': False,
        }
        
        st_dir_path = os.path.join(sub_path, tract_folder)
        st_dir = os.listdir(st_dir_path)
        params_exp['tractogram_path'] = [os.path.join(st_dir_path, f) for f in st_dir]

        origdata = VoxCoordDataLoader(params_exp)
        dwi = origdata.get_spherical_harmonics_coefficients(
            origdata.dwi, origdata.bvals, origdata.bvecs, subname='other', sh_order=6
        )
        l, w, h, c = dwi.shape
        dwi_padded = np.zeros((l + 8, w + 8, h + 8, c))
        dwi_padded[4:-4, 4:-4, 4:-4, :] = dwi
        dwi_list.append(dwi_padded)

        # 构建 3D 坐标映射图
        lp, wp, hp, _ = dwi_padded.shape
        dwi_coord_map = np.array(list(np.ndindex(lp, wp, hp))).reshape(lp, wp, hp, 3)
        coord_dwi_list.append(dwi_coord_map)

        sample_size = len(origdata.arysq)
        st_list.append(origdata.arysq.copy())
        stbn_list.append(origdata.bundle_name)

        total_samples += sample_size
        cum_lens.append(total_samples)

    return dwi_list, coord_dwi_list, st_list, stbn_list, cum_lens

class TractographyDataset(data.Dataset):
    """通用且支持动态受试者数量的 Dataset 类"""
    def __init__(self, dwi_list, st_list, cum_lens, coord_dwi_list, stbn_list, balance=False, loss_r=None, block_size=96):
        self.dwi_list = dwi_list
        self.st_list = st_list
        self.cum_lens = cum_lens
        self.coord_dwi_list = coord_dwi_list
        self.stbn_list = stbn_list
        self.balance = balance
        self.loss_r = loss_r
        self.block_size = block_size

    def __len__(self):
        return self.cum_lens[-1] if self.cum_lens else 0

    def __getitem__(self, index):
        nb_volume = bisect.bisect_right(self.cum_lens, index)
        if nb_volume == 0:
            idx = index
        else:
            idx = index - self.cum_lens[nb_volume - 1]

        x, y, v = convert2dwi_patch(
            self.dwi_list[nb_volume],
            self.coord_dwi_list[nb_volume],
            self.st_list[nb_volume][idx].copy(),
            self.block_size
        )
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        v = torch.from_numpy(v).to(torch.float32)

        if self.balance and self.loss_r is not None:
            b_n = self.stbn_list[nb_volume][idx]
            if b_n in self.loss_r and b_n not in ['FX_R', 'FX_L']:
                l_r = self.loss_r[b_n]
            else:
                l_r = max(self.loss_r.values())
            l_r = torch.from_numpy(np.asarray(l_r)).to(torch.float32)
            return x, y, v, l_r
        else:
            return x, y, v

@torch.no_grad()
def estimate_loss(model, dataloaders, eval_iters, device, ctx, balance):
    out = {}
    model.eval()
    for split, loader in dataloaders.items():
        if loader is None:
            continue
        max_steps = eval_iters.get(split, 100)
        losses = []
        for i, batch in enumerate(loader):
            if i >= max_steps:
                break
            if balance:
                X, Y, V, L_R = [x.to(device) for x in batch]
                logits, loss = model(X, Y, V, L_R)
            else:
                X, Y, V = [x.to(device) for x in batch]
                logits, loss = model(X, Y, V)
            losses.append(loss.item())
        out[split] = np.mean(losses) if losses else 0.0
    model.train()
    return out

def main():
    args = parse_args()

    # DDP 初始化
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = int(os.environ['RANK']) == 0
        seed_offset = int(os.environ['RANK'])
    else:
        master_process = True
        seed_offset = 0
        device = args.device

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.float32)

    # 类别平衡设置
    balance = args.balance_file != 'None' and os.path.exists(args.balance_file)
    loss_r = np.load(args.balance_file, allow_pickle=True).item() if balance else None

    # 打包文件名格式模板
    fn_templates = {
        'dwi': args.dwi_template,
        'wm_mask': args.wm_mask_template,
        'bval': args.bval_template,
        'bvec': args.bvec_template,
        'tractogram_folder': args.tractogram_folder
    }

    # 根据传入的配置动态加载数据
    tr_dwi, tr_coord, tr_st, tr_stbn, tr_lens = load_subject_data(args.train_subj, args.train_subpath, args.data_root, fn_templates)
    va_dwi, va_coord, va_st, va_stbn, va_lens = load_subject_data(args.valid_subj, args.valid_subpath, args.data_root, fn_templates)
    te_dwi, te_coord, te_st, te_stbn, te_lens = load_subject_data(args.test_subj, args.test_subpath, args.data_root, fn_templates)

    train_set = TractographyDataset(tr_dwi, tr_st, tr_lens, tr_coord, tr_stbn, balance, loss_r, args.block_size)
    val_set = TractographyDataset(va_dwi, va_st, va_lens, va_coord, va_stbn, balance, loss_r, args.block_size)
    test_set = TractographyDataset(te_dwi, te_st, te_lens, te_coord, te_stbn, balance, loss_r, args.block_size)

    loader_args = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}
    dataloaders = {
        'train': data.DataLoader(train_set, **loader_args),
        'val': data.DataLoader(val_set, **loader_args),
        'test': data.DataLoader(test_set, **loader_args)
    }

    # 初始化模型
    gptconf = GPTConfig(
        n_layer=6, n_head=6, n_embd=192,
        block_size=args.block_size, bias=False, dropout=0.1
    )
    model = GPT.from_pretrained('gpt2', override_args=dict(dropout=0.1))
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=args.lr, betas=(0.9, 0.95), device_type=device_type)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model
    best_val_loss = 1e9
    iter_num = 0

    if args.wandb_run_name and master_process:
        import wandb
        wandb.init(project='owt', name=args.wandb_run_name, config=vars(args))

    # 训练循环
    for epoch in range(args.num_epochs):
        for batch in dataloaders['train']:
            if iter_num % args.eval_interval == 0 and master_process:
                losses = estimate_loss(model, dataloaders, {'train': 100, 'val': 100, 'test': 20}, device, ctx, balance)
                print(f"Step {iter_num}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}, Test Loss {losses['test']:.4f}")
                
                if wandb.run:
                    wandb.log({"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'], "test/loss": losses['test']})

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                        }
                        torch.save(checkpoint, os.path.join(args.out_dir, args.ckpt_sn))

            with ctx:
                if balance:
                    X, Y, V, L_R = [x.to(device) for x in batch]
                    _, loss = model(X, Y, V, L_R)
                else:
                    X, Y, V = [x.to(device) for x in batch]
                    _, loss = model(X, Y, V)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            iter_num += 1

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()