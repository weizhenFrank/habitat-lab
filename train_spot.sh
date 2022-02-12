#!/bin/bash
#SBATCH --job-name=spot_kinematic_hm3d_gibson
#SBATCH --output=spot_kinematic_hm3d_gibson.out
#SBATCH --error=spot_kinematic_hm3d_gibson.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition long
#SBATCH --cpus-per-task=7
#SBATCH --exclude eva
#SBATCH --chdir /coc/testnvme/jtruong33/google_nav/habitat-lab

export NCCL_LL_THRESHOLD=0
export CUDA_VISIBLE_DEVICES=0

srun /nethome/jtruong33/miniconda3/envs/habitat-outdoor/bin/python -u -m habitat_baselines.run \
    --exp-config /coc/testnvme/jtruong33/google_nav/habitat-lab/habitat_baselines/config/pointnav/ddppo_spotnav.yaml \
    --run-type train
