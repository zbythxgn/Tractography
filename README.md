# Deep Learning-Based Diffusion MRI Tractography: Integrating Spatial and Anatomical Information

本项目基于 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 进行深度二次开发，将 Transformer 架构（GPT）应用于扩散磁共振diffusion MRI（dMRI）的白质纤维束追踪（Tractography）。通过将连续的 3D DWI Patch 序列输入 GPT 模型，实现高精度的纤维生长方向预测与流线（Streamlines）追踪。

---

## 🌟 主要特性

- **基于 GPT 的序列建模**：利用自注意力机制捕获纤维束在三维空间中的长程上下文依赖。
- **3D 卷积与 DWI Patch 提取**：结合 3D 卷积特征提取网络与三线性插值，实时提取邻域 DWI 信号。
- **双向追踪（Forward & Backward Tracking）**：支持正向与反向联合生长追踪，自动处理边界停止条件与曲率限制。
- **高度可配置**：支持多受试者并行训练、类别平衡损失（Balanced Cosine Loss）以及分布式 DDP 训练。

---

## 🛠️ 环境准备与依赖项 (Dependencies)

项目运行依赖 PyTorch 及专用于扩散磁共振数据处理的神经影像学库（如 `scilpy` 与 `dipy`）。

### 核心依赖

- **Python** >= 3.8
- **PyTorch** >= 2.0 (推荐支持 CUDA 及 FlashAttention)
- **dipy** (用于球谐函数拟合、坐标变换及流线计算)
- **scilpy** (用于 Tractography 数据预处理及分析)
- **nibabel** (用于 NIfTI 及 TCK 文件读写)
- **numpy**, **scipy**, **wandb** (可选，用于实验日志记录)

### 虚拟环境安装

可通过项目根目录提供的 `environment.yml` 一键创建并激活运行环境：

```bash
conda env create -f environment.yml -n myenv
conda activate myenv 
```


## 📁 数据集组织格式 (Dataset Structure)
训练与测试数据需按照受试者（Subject）目录进行组织，结构示例如下：
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

## 🚀 快速开始 (Usage)
### 1. 模型训练 (Training)
训练逻辑封装在 `train.py` 中，建议通过启动脚本 `train.sh`执行：
```bash train.sh```

关键配置参数说明 (`train.sh`)：
```
--data_root：数据集根目录路径。

--train_subj / --valid_subj：指定训练与验证集受试者编号。

--dwi_template / --wm_mask_template：指定 DWI 文件及白质 Mask 的相对路径模板。

--batch_size：每卡 Batch 大小（默认 256）。

--block_size：GPT 上下文序列长度（默认 96）。

--ckpt_sn：模型权重保存名称。
```

### 2. 纤维束追踪与测试 (Tracking & Inference)
训练完成后，使用 `track.sh` 调用 `track.py` 开展纤维束推理与生成 .tck 文件：
```bash track.sh```

关键配置参数说明 (`track.sh`)：

--dwi_path / --bvec / --bval：待追踪受试者的 DWI 图像及 b-values/b-vectors 文件。

--tracking_mask：白质追踪掩模（WM tracking mask）。

--seeding_mask：种子点生成掩模（Seeding mask）。

--ckpt_path：训练好的模型权重文件路径 (.pt)。

--seeds_per_vox：每个体素生成的种子点数量。

--batchsize：并行追踪的流线 Batch 数量（默认 2048）。

--out_dir 与 --out_tck：输出 .tck 文件的保存路径及后缀。

## 💡 代码架构 (Codebase Structure)
`model.py`：定义结合 3D 特征提取器的 GPT 模型架构与 masked/balanced 余弦相似度损失函数。

`train.py`：数据集加载、球谐函数（SH）特征转换及基于 DDP 的模型训练主循环。

`track.py`：实现 Tracker 与 BackwardTracker 类，包含 GPU 加速的 Patch 采样与双向纤维生长控制。

`VoxCoordDataLoader.py`：体素坐标与 DWI 数据转换加载器。

`train.sh` / `track.sh`：快速启动训练与评估的一键 Bash 脚本