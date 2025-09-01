#!/bin/bash

#SBATCH --job-name=jax_h200_offline
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH -p mit_normal_gpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH -o notebook_%A.out

set -euo pipefail

# Initialize environment modules
source "$HOME/anaconda3/bin/activate" jax_cuda_offline
export PYTHONNOUSERSITE=1

# Keep ONLY the env's libraries visible (cuDNN from conda-forge)
unset LD_LIBRARY_PATH CUDA_HOME CUDNN_HOME
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

# NVCC wheel (ptxas + libdevice) for XLA toolchain
export NVCC_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export PATH="$NVCC_DIR/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVCC_DIR ${XLA_FLAGS:-}"

# JAX runtime
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_PLATFORMS=cuda
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Set the IP address and port
IP=`hostname -i`
PORT=`shuf -i 2000-65000 -n 1`

# Run Jupyter notebook
$HOME/anaconda3/bin/jupyter notebook --ip=$IP --port=$PORT --no-browser
