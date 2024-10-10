#!/bin/bash
#SBATCH --job-name=diff_hef     # Job name
#SBATCH --nodes=5             # Number of nodes
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=long      # Partition name
#SBATCH --output=multi_gpu_%j.out     # Standard output and error log
#SBATCH --error=multi_gpu_%j.err      # Error log

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


module --quiet purge

module load cuda/11.8/cudnn/8.6
module load anaconda/3

# Activate pre-existing environment.
source  activate harmonic
cd $HOME/scratch/HarmonicExponentialBayesFitler/diff_filter/

wandb login 8ffe865c4b82a4e1f84ebcf8cc9681892e828854


unset CUDA_VISIBLE_DEVICES

# Execute Python script
wandb agent korra141/differential-hef/5f04l9ae
python differentiable_filters/example_training_code/run_example.py
