#! /bin/bash
# Benchmark the original and optimized tracking implementations sequentially.
# Run with: nohup bash example_test_2.sh >/dev/null 2>&1 &
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/yyq/.conda/envs/ft/bin/python"
time_log="${script_dir}/time.log"
DATA_ROOT="/path/to/tracking_data"
sub="1006"
# GPU 1 was idle when this benchmark was launched; change this if needed.
device="cuda:2"
dwi_path="${DATA_ROOT}/sub-${sub}/dwi/sub-${sub}__dwi.nii.gz"
bvec="${DATA_ROOT}/sub-${sub}/dwi/sub-${sub}__dwi.bvec"
bval="${DATA_ROOT}/sub-${sub}/dwi/sub-${sub}__dwi.bval"
tracking_mask="${DATA_ROOT}/sub-${sub}/mask/sub-${sub}__mask_wm.nii.gz"
seeding_mask="${tracking_mask}"
ckpt_path="${script_dir}/ckpt_example.pt"
seeds_per_vox=5
batchsize=2048
use_seeding_mask=False
out_dir="/data/d1/YYQ/Tractography/"

printf 'Tracking timing benchmark started: %s\n' "$(date -Is)" > "${time_log}"
printf 'device=%s, subject=%s, batch_size=%s\n' "${device}" "${sub}" "${batchsize}" >> "${time_log}"

run_case() {
    local sample_script="$1"
    local label="$2"
    local postfix="${sub}_45d_${seeds_per_vox}seeds_${label}.tck"
    local run_log="${script_dir}/.${label}_timing_run.log"

    printf '\n=== %s (%s) ===\n' "${label}" "${sample_script}" | tee -a "${time_log}"
    if "${python_bin}" "${script_dir}/${sample_script}" \
        --device "${device}" \
        --sub "${sub}" \
        --dwi_path "${dwi_path}" \
        --bvec "${bvec}" \
        --bval "${bval}" \
        --tracking_mask "${tracking_mask}" \
        --seeding_mask "${seeding_mask}" \
        --ckpt_path "${ckpt_path}" \
        --seeds_per_vox "${seeds_per_vox}" \
        --out_dir "${out_dir}" \
        --out_tck "${postfix}" \
        --batchsize "${batchsize}" \
        --use_seeding_mask "${use_seeding_mask}" \
        > "${run_log}" 2>&1; then
        grep -E '^(FIRST_DIRECTION_STEP_SECONDS|FULL_TRACKING_SECONDS)=' "${run_log}" | tee -a "${time_log}"
        printf 'status=completed\n' | tee -a "${time_log}"
    else
        status=$?
        printf 'status=failed (exit=%s); details=%s\n' "${status}" "${run_log}" | tee -a "${time_log}"
        tail -n 80 "${run_log}" >> "${time_log}"
        return "${status}"
    fi
}

run_case "track.py" "v1"

printf '\nTracking timing benchmark finished: %s\n' "$(date -Is)" | tee -a "${time_log}"