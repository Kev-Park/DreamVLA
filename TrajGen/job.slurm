#!/bin/bash
#SBATCH --job-name=gen_traj     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G         # memory total (4GB good for traj refinement)
#SBATCH --gres=gpu:1             # number of GPUs per node
#SBATCH --constraint=gpu80
#SBATCH --time=3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=kp0374@princeton.edu

module purge
module load anaconda3/2025.6
#conda activate omnicontrol

export CUDA_VISIBLE_DEVICES=0

bash collect_Pick.sh
#bash refine_Pick_al.sh

