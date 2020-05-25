#/bin/bash

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}"
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
export PYTHONPATH="${PYTHONPATH}:/srv/share3/jtruong33/develop/sim2real/map_and_plan_agent"

MODEL_PATH="/srv/share3/jtruong33/develop/sim2real/data/new_checkpoints/ddppo_pointnav_depth_noise_goal_regress_0.15/ckpt.000000028.7.198153.pth"

SENSORS="RGB_SENSOR,DEPTH_SENSOR"
#SENSORS="DEPTH_SENSOR"
BACKBONE="resnet50"
HIDDEN_SIZE=512
NUM_RECURRENT_LAYERS=2
NORMALIZE_VISUAL_INPUTS=1
SPL_MAX_COLLISIONS="40"

# EPISODE_DATASET_PATH="/srv/share3/jtruong33/develop/sim2real/data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
# EPISODE_DATASET_SPLIT="val"

EPISODE_DATASET_PATH="/srv/share3/jtruong33/develop/sim2real/data/datasets/pointnav/coda/{split}/{split}.json.gz"
EPISODE_DATASET_SPLIT=$1
RUN=$2
VIDEO_OPTION="['disk']"
VIDEO_DIR="videos/finetune/262_all/${EPISODE_DATASET_SPLIT}_${RUN}"

python -u evaluation/evaluate_simulation_coda.py \
    --model-path ${MODEL_PATH} \
    --data-split ${EPISODE_DATASET_SPLIT} \
    --sensors ${SENSORS} \
    --hidden-size ${HIDDEN_SIZE} \
    --normalize-visual-inputs ${NORMALIZE_VISUAL_INPUTS} \
    --backbone ${BACKBONE} \
    --num-recurrent-layers ${NUM_RECURRENT_LAYERS} \
    --noisy\
    "TEST_EPISODE_COUNT" "5" \
    "TASK_CONFIG.TASK.SPL.MAX_COLLISIONS" ${SPL_MAX_COLLISIONS} \
    "TASK_CONFIG.DATASET.DATA_PATH" ${EPISODE_DATASET_PATH} \
    "TASK_CONFIG.DATASET.SPLIT" ${EPISODE_DATASET_SPLIT} \
    "VIDEO_OPTION" ${VIDEO_OPTION} \
    "VIDEO_DIR" ${VIDEO_DIR} \
