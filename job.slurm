#!/bin/bash
#SBATCH --job-name=rl_train     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of GPUs per node
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=kp0374@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/kp0374/conda/dreamcontrol

export CUDA_VISIBLE_DEVICES=0

cd Training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --kit_args "--/persistent/isaac/asset_root/default=/scratch/network/kp0374/isaacsimassets/Assets/Isaac/4.5" --task=Isaac-Motion-Tracking-Pick-v0 --headless --device cuda:0 