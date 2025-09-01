#!/bin/bash

#SBATCH --job-name=jax_h200_local
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH -p mit_normal_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH -o notebook_%A.out

set -euo pipefail

# Initialize environment modules
source /etc/profile.d/modules.sh
module purge
module use /orcd/software/core/001/modulefiles
module load cuda/12.4.0

source "$HOME/anaconda3/bin/activate" jax_cuda_local
export PYTHONNOUSERSITE=1

export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME ${XLA_FLAGS:-}"

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
