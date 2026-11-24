#!/usr/bin/env bash
#SBATCH --job-name=dsflash-psr
#SBATCH --nodelist=labserver0
#SBATCH --gres=gpu:A6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
set -euo pipefail

source_dir=$1
run_dir=$2
dataset=$3
manifest=$4
config=$5
shift 5

mkdir -p "$run_dir"
exec 9>"$run_dir/train.lock"
flock -n 9
cd "$source_dir"

storage=$(dirname "$dataset")
export HF_HOME="$storage/hf_cache" HUGGINGFACE_HUB_CACHE="$storage/hf_cache"
export TORCH_HOME="$storage/torch_cache" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 TZ=Asia/Shanghai

active_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$active_gpu_pids" ]]; then
    printf 'Existing GPU compute processes: %s\n' "$active_gpu_pids" >&2
    exit 75
fi

python_bin="${KAF_PYTHON:-$HOME/miniconda3/envs/kafnet/bin/python}"
"$python_bin" -c 'import torch; assert torch.cuda.device_count() == 4, torch.cuda.device_count()'
echo "Starting DSFlash on four GPUs at $(date -Is)"
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m experiments.dsflash_psr train --config "$config" --run-dir "$run_dir" \
    --data-dir "$dataset" --manifest "$manifest" "$@"
