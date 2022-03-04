#!/bin/bash
#SBATCH --job-name=$TEMPLATE
#SBATCH --output=$LOG.out
#SBATCH --error=$LOG.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 7
#SBATCH --partition $PARTITION

#SBATCH --chdir $HABITAT_REPO_PATH
export CUDA_LAUNCH_BLOCKING=1
srun /nethome/jtruong33/miniconda3/envs/igib_dyn/bin/python -u -m habitat_baselines.run \
    --exp-config $CONFIG_YAML \
    --run-type eval