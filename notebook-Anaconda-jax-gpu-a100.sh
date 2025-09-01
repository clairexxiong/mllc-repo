#!/bin/bash

#SBATCH --job-name=jax_sloangpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH -p sched_mit_sloan_gpu_r8
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH -o notebook_%A.out

set -euo pipefail

# Initialize environment modules
source /etc/profile.d/modules.sh
source "$HOME/anaconda3/bin/activate" jax_gpu124

module purge
module use /orcd/software/core/001/modulefiles
module load cuda/12.4.0

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH now: $LD_LIBRARY_PATH"

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
