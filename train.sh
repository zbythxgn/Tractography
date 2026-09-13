#!/bin/bash

PYTHON_BIN="/bin/python"
SCRIPT_PATH="./train.py"

# 数据根路径与输出路径
DATA_ROOT="/path/to/example_data"
BALANCE_FILE='None'
OUT_DIR="/path/to/save/"

# 各分集的相对存储路径
TRAIN_SUBPATH="trainset"
VALID_SUBPATH="validset"
TEST_SUBPATH="testset"

# 受试者数据列表
TRAIN_SUBJS="sub-1006"
VALID_SUBJS="sub-1006"
TEST_SUBJS="sub-1006"

# 文件名格式模板 (使用 {sub} 作为受试者编号占位符)
DWI_TEMPLATE="dwi/{sub}__dwi.nii.gz"
WM_MASK_TEMPLATE="mask/{sub}__mask_wm.nii.gz"
BVAL_TEMPLATE="dwi/{sub}__dwi.bval"
BVEC_TEMPLATE="dwi/{sub}__dwi.bvec"
TRACTOGRAM_FOLDER="1mm-tractogram"

${PYTHON_BIN} ${SCRIPT_PATH} \
    --device 'cuda:0' \
    --data_root "${DATA_ROOT}" \
    --balance_file "${BALANCE_FILE}" \
    --out_dir "${OUT_DIR}" \
    --train_subpath "${TRAIN_SUBPATH}" \
    --valid_subpath "${VALID_SUBPATH}" \
    --test_subpath "${TEST_SUBPATH}" \
    --train_subj ${TRAIN_SUBJS} \
    --valid_subj ${VALID_SUBJS} \
    --test_subj ${TEST_SUBJS} \
    --dwi_template "${DWI_TEMPLATE}" \
    --wm_mask_template "${WM_MASK_TEMPLATE}" \
    --bval_template "${BVAL_TEMPLATE}" \
    --bvec_template "${BVEC_TEMPLATE}" \
    --tractogram_folder "${TRACTOGRAM_FOLDER}" \
    --ckpt_sn 'ckpt_example.pt' \
    --batch_size 256 \
    --num_epochs 50 \
    --block_size 96 \
    --lr 6e-4 \
    --num_workers 16 \
    --eval_interval 1000