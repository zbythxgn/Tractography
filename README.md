# Deep Learning-Based Diffusion MRI Tractography: Integrating Spatial and Anatomical Information

This project presents a deep learning framework for white matter streamline tractography in diffusion magnetic resonance imaging (dMRI), substantially adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
---

## 🌟 Key Features

- **GPT-Based Sequence Modeling:**：Utilizes self-attention mechanisms to capture long-range spatial and contextual dependencies along fiber trajectories in 3D space.
- **3D Convolution & Dynamic Patch Extraction**：Combines a 3D convolutional network for feature extraction with trilinear interpolation to dynamically sample localized neighborhood DWI signals.
- **Bidirectional Tractography**：Supports joint forward and backward streamline propagation while enforcing termination criteria and curvature constraints.
- **Highly Configurable**：Features multi-subject parallel training, Balanced Cosine Loss, and Distributed Data Parallel (DDP) multi-GPU execution.

---


## 🛠️ Dependencies & Environment Setup

The codebase relies on PyTorch alongside specialized neuroimaging libraries for diffusion MRI processing (such as `scilpy` and `dipy`). 

### Core Dependencies

- **Python** >= 3.8
- **PyTorch** >= 2.0 (Recommended with CUDA and FlashAttention support)
- **dipy** (For spherical harmonics fitting, coordinate transformations, and streamline operations)
- **scilpy** (For tractography preprocessing and analytical utilities)
- **nibabel** (For NIfTI and TCK file I/O operations)
- **numpy**, **scipy**, **wandb** (Optional, for experiment logging)

### Environment Installation

You can recreate and activate the environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml -n myenv
conda activate myenv 
```


## 📁 Dataset Organization
Training and evaluation datasets must be organized by subject directories following this structure:
```
data_root/
├── trainset/
│   ├── sub-1001/
│   │   ├── dwi/
│   │   │   ├── sub-1001__dwi.nii.gz
│   │   │   ├── sub-1001__dwi.bval
│   │   │   └── sub-1001__dwi.bvec
│   │   ├── mask/
│   │   │   └── sub-1001__mask_wm.nii.gz
│   │   └── 1mm-tractogram/
│   │       └── *.tck
└── testset/
    └── sub-1006/
        ...
```


## 📌 Important Notes & Best Practices

To achieve optimal training stability and tractography performance, observe the following guidelines:

1. **Spatial Resolution Consistency**：The model performs optimally when training and evaluation datasets share the same spatial resolution.
2. **Streamline Preprocessing**：
   - Ground-truth (GT) streamlines in the training set should be resampled to a **fixed step-size**.
   - Streamline sequences must be **truncated or zero-padded to 96 timesteps** (corresponding to the model's default `block_size`).

---


## 🚀 Usage
### 1. Model Training
The core training logic is encapsulated in `train.py` . Execution via the wrapper script `train.sh`is recommended:
```bash train.sh```

Key Parameters (`train.sh`)：
```
--data_root: Path to the root directory of the dataset.

--train_subj / --valid_subj: Subject IDs assigned for training and validation splits.

--dwi_template / --wm_mask_template: Relative path templates for DWI volumes and white matter masks.

--batch_size: Per-GPU batch size (default: 256).

--block_size: Maximum context sequence length of the Transformer (default: 96).

--ckpt_sn: Checkpoint identifier for saving model weights.
```

### 2. Tractography & Inference
Upon completing training, execute `track.sh` (which invokes `track.py`) to perform streamline inference and generate `.tck` files:
```bash track.sh```

Key Parameters (`track.sh`)：
```
--dwi_path / --bvec / --bval: Paths to the subject's DWI volume, b-vectors, and b-values.

--tracking_mask: Binary white matter tracking mask.

--seeding_mask: Binary mask defining seeding regions.

--ckpt_path: Path to the trained checkpoint file (.pt).

--seeds_per_vox: Number of seed points generated per seed voxel.

--batchsize: Number of streamlines processed concurrently during tracking (default: 2048).

--out_dir & --out_tck: Output directory and filename suffix for generated .tck files.
```

## 💡 Codebase Architecture
`model.py`: Defines the Transformer decoder architecture integrated with a 3D spatial feature extractor, alongside masked and balanced cosine similarity loss functions.

`train.py`: Handles dataset loading, Spherical Harmonics (SH) feature transformations, and the DDP-based training loop.

`track.py`: Implements the Tracker and BackwardTracker classes, providing GPU-accelerated patch sampling and bidirectional fiber propagation control.

`VoxCoordDataLoader.py`: Data loader managing transformations between voxel coordinates and DWI signals.

`train.sh` / `track.sh`: Bash wrapper scripts for execution of training and evaluation pipelines.