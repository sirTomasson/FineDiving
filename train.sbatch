#!/bin/bash
# parameters for slurm
#SBATCH -c 8                       # number of cores, 1
#SBATCH --gres=gpu:ampere:1
#SBATCH --mail-type=END,FAIL          # email status changes (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=7-00:00:00

# show actual node in output file, useful for diagnostics
hostname

nvidia-smi

# Load all required software modules, including Python
module load nvidia/cuda-11.8
module load python/3.10.7   # Load the desired Python module, version might vary

# It's nice to have some information logged for debugging
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128

# Run the Python script
bash train.sh TSA FineDiving 0,1
