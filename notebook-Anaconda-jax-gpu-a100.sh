#!/bin/bash

#SBATCH --job-name=condajax
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH -p sched_mit_sloan_gpu_r8
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH -o notebook_%A.out

# Initialize environment modules
source /etc/profile.d/modules.sh
source "$HOME/anaconda3/bin/activate" jax_gpu124

module purge
module use /orcd/software/core/001/modulefiles
module load cuda/12.4.0
# module load cudnn/9.8.0.87-cuda12

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH now: $LD_LIBRARY_PATH"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Set the IP address and port
IP=`hostname -i`
PORT=`shuf -i 2000-65000 -n 1`

# Run Jupyter notebook
$HOME/anaconda3/bin/jupyter notebook --ip=$IP --port=$PORT --no-browser
