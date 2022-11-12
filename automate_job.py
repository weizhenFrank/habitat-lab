"""
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

automate_command = "python " + " ".join(sys.argv)
BASE_DIR = "/coc/testnvme/jtruong33/google_nav_ver/habitat-lab"
HABITAT_LAB = os.path.join(BASE_DIR, "habitat-lab/habitat")
HABITAT_BASELINES = os.path.join(BASE_DIR, "habitat-baselines/habitat_baselines")
CONDA_ENV = "/nethome/jtruong33/miniconda3/envs/outdoor-ver/bin/python"
RESULTS = "/coc/testnvme/jtruong33/results/outdoor_ver_results"
SLURM_TEMPLATE = os.path.join(BASE_DIR, "slurm_template.sh")
EVAL_SLURM_TEMPLATE = os.path.join(BASE_DIR, "eval_slurm_template.sh")

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")

# Training
parser.add_argument("-sd", "--seed", type=int, default=1)
parser.add_argument("-r", "--robot", default="Spot")
parser.add_argument("-p", "--partition", default="long")
parser.add_argument("-ds", "--dataset", default="ny")
parser.add_argument("--constraint", default="x")

# Policy
parser.add_argument("--policy-name", default="PointNavResNetPolicy")
parser.add_argument("-rs", "--robot-scale", type=float, default=1.0)
parser.add_argument("-mr", "--map-resolution", type=int, default=100)
parser.add_argument("-ne", "--num-environments", type=int, default=16)

# Evaluation
parser.add_argument("-e", "--eval", default=False, action="store_true")
parser.add_argument("-cpt", "--ckpt", type=int, default=-1)
parser.add_argument("-v", "--video", default=False, action="store_true")
parser.add_argument("-d", "--debug", default=False, action="store_true")
parser.add_argument("-x", default=False, action="store_true")
parser.add_argument("--ext", default="")
args = parser.parse_args()

EXP_YAML = "config/pointnav/ddppo_pointnav_spot.yaml"
TASK_YAML = "config/tasks/pointnav_spot.yaml"

experiment_name = args.experiment_name

dst_dir = os.path.join(RESULTS, experiment_name)
eval_dst_dir = os.path.join(RESULTS, experiment_name, "eval", args.control_type)

exp_yaml_path = os.path.join(HABITAT_BASELINES, EXP_YAML)
task_yaml_path = os.path.join(HABITAT_LAB, TASK_YAML)

new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_exp_yaml_path = os.path.join(dst_dir, os.path.basename(exp_yaml_path))

exp_name = f"_{args.control_type}"

if args.eval:
    exp_name += "_eval"
if args.ckpt != -1:
    exp_name += f"_ckpt_{args.ckpt}"
    eval_dst_dir += f"_ckpt_{args.ckpt}"
if args.video:
    exp_name += "_video"
    eval_dst_dir += "_video"
if args.ext != "":
    exp_name += "_" + args.ext
    eval_dst_dir += "_" + args.ext
if args.dataset != "hm3d_gibson":
    exp_name += f"_{args.dataset}"
    eval_dst_dir += f"_{args.dataset}"

new_eval_task_yaml_path = (
    os.path.join(eval_dst_dir, os.path.basename(task_yaml_path)).split(".yaml")[0]
    + exp_name
    + ".yaml"
)
new_eval_exp_yaml_path = (
    os.path.join(eval_dst_dir, os.path.basename(exp_yaml_path)).split(".yaml")[0]
    + exp_name
    + ".yaml"
)

if args.robot_scale == 0.1:
    robot_urdf = "/coc/testnvme/jtruong33/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid_rot_fix_0.1x.urdf"

# Training
if not args.eval:
    # Create directory
    if os.path.isdir(dst_dir):
        response = input(f"'{dst_dir}' already exists. Delete or abort? [d/A]: ")
        if response == "d":
            print(f"Deleting {dst_dir}")
            shutil.rmtree(dst_dir)
        else:
            print("Aborting.")
            exit()
    os.mkdir(dst_dir)
    print("Created " + dst_dir)

    with open(os.path.join(dst_dir, "automate_job_cmd.txt"), "w") as f:
        f.write(automate_command)
    print("Saved automate command: " + os.path.join(dst_dir, "automate_job_cmd.txt"))

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        task_yaml_data = f.read().splitlines()

    # robots_heights = [robot_heights_dict[robot] for robot in robots]

    for idx, i in enumerate(task_yaml_data):
        if i.startswith("    robot:"):
            task_yaml_data[idx] = f"    robot: '{args.robot}'"
        elif i.startswith(
            "      positon: [ -0.03740343144695029, 0.5, -0.4164822634134684 ]"
        ):
            position = (
                np.array([-0.03740343144695029, 0.5, -0.4164822634134684])
                * args.robot_scale
            )
            task_yaml_data[idx] = f"      positon: {list(position)}"
        elif i.startswith(
            "      positon: [ 0.03614789234067159, 0.5, -0.4164822634134684 ]"
        ):
            position = (
                np.array([0.03614789234067159, 0.5, -0.4164822634134684])
                * args.robot_scale
            )
            task_yaml_data[idx] = f"      positon: {list(position)}"
        elif i.startswith("      max_depth:"):
            max_depth = 3.5 * args.robot_scale
            task_yaml_data[idx] = f"      max_depth: {max_depth:.2f}"
        elif i.startswith("        robot_urdf:"):
            task_yaml_data[idx] = f"        robot_urdf: {robot_urdf}"
        elif i.startswith("      map_resolution:"):
            task_yaml_data[idx] = f"      map_resolution: {args.map_resolution}"
            if args.rotate_map:
                task_yaml_data[idx] = f"    ROTATE_MAP: True"
        elif i.startswith("        lin_vel_range:"):
            lin_vel = 0.5 * args.robot_scale
            task_yaml_data[idx] = f"        lin_vel_range: [ -{lin_vel}, {lin_vel} ]"
        elif i.startswith("        min_rand_pitch:"):
            task_yaml_data[idx] = f"        min_rand_pitch: {args.randomize_pitch_min}"
        elif i.startswith("        max_rand_pitch:"):
            task_yaml_data[idx] = f"        max_rand_pitch: {args.randomize_pitch_max}"
        elif i.startswith("  seed:"):
            task_yaml_data[idx] = f"  seed: {args.seed}"
        elif i.startswith("    data_path:"):
            if args.dataset == "ny":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/{split}/{split}.json.gz"
            task_yaml_data[idx] = f"    data_path: {data_path}"
        elif i.startswith("      noise_multiplier:"):
            task_yaml_data[idx] = f"      noise_multiplier: {args.noise_percent}"
    with open(new_task_yaml_path, "w") as f:
        f.write("\n".join(task_yaml_data))
    print("Created " + new_task_yaml_path)

    # Create experiment yaml file, using file within Habitat Lab repo as a template
    with open(exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(exp_yaml_data):
        if i.startswith("  base_task_config_path:"):
            exp_yaml_data[idx] = f"  base_task_config_path: '{new_task_yaml_path}'"
        elif i.startswith("  tensorboard_dir:"):
            exp_yaml_data[
                idx
            ] = f"  tensorboard_dir:    '{os.path.join(dst_dir, 'tb')}'"
        elif i.startswith("  num_environments:"):
            exp_yaml_data[idx] = f"  num_environments: {args.num_environments}"
            if "ferst" in args.dataset or "coda" in args.dataset:
                exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
        elif i.startswith("  video_dir:"):
            exp_yaml_data[
                idx
            ] = f"  video_dir:          '{os.path.join(dst_dir, 'video_dir')}'"
        elif i.startswith("  eval_ckpt_path_dir:"):
            exp_yaml_data[
                idx
            ] = f"  eval_ckpt_path_dir: '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("  checkpoint_folder:"):
            exp_yaml_data[
                idx
            ] = f"  checkpoint_folder:  '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("  txt_dir:"):
            exp_yaml_data[
                idx
            ] = f"  txt_dir:            '{os.path.join(dst_dir, 'txts')}'"
    with open(new_exp_yaml_path, "w") as f:
        f.write("\n".join(exp_yaml_data))
    print("Created " + new_exp_yaml_path)

    # Create slurm job
    with open(SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace("$TEMPLATE", experiment_name)
        slurm_data = slurm_data.replace("$CONDA_ENV", CONDA_ENV)
        slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", HABITAT_LAB)
        slurm_data = slurm_data.replace("$LOG", os.path.join(dst_dir, experiment_name))
        slurm_data = slurm_data.replace("$CONFIG_YAML", new_exp_yaml_path)
        slurm_data = slurm_data.replace("$PARTITION", args.partition)
    if args.debug:
        slurm_data = slurm_data.replace("$GPUS", "1")
    else:
        slurm_data = slurm_data.replace("$GPUS", f"{args.num_gpus}")
    if args.constraint == "6000_a40":
        slurm_data = slurm_data.replace(
            "# CONSTRAINT", "#SBATCH --constraint rtx_6000|a40"
        )
    elif args.constraint == "6000":
        slurm_data = slurm_data.replace("# CONSTRAINT", "#SBATCH --constraint rtx_6000")
    elif args.constraint == "a40":
        slurm_data = slurm_data.replace("# CONSTRAINT", "#SBATCH --constraint a40")
    slurm_path = os.path.join(dst_dir, experiment_name + ".sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_data)
    print("Generated slurm job: " + slurm_path)

    if not args.x:
        # Submit slurm job
        cmd = "sbatch " + slurm_path
        subprocess.check_call(cmd.split(), cwd=dst_dir)
    else:
        print(slurm_data)

    print(
        f"\nSee output with:\ntail -F {os.path.join(dst_dir, experiment_name + '.err')}"
    )

# # Evaluation
# else:
#     # Make sure folder exists
#     assert os.path.isdir(dst_dir), f"{dst_dir} directory does not exist"
#     os.makedirs(eval_dst_dir, exist_ok=True)
#     with open(os.path.join(eval_dst_dir, "automate_job_cmd.txt"), "w") as f:
#         f.write(automate_command)
#     print(
#         "Saved automate command: " + os.path.join(eval_dst_dir, "automate_job_cmd.txt")
#     )
#
#     # Create task yaml file, using file within Habitat Lab repo as a template
#     with open(task_yaml_path) as f:
#         eval_yaml_data = f.read().splitlines()
#
#     for idx, i in enumerate(eval_yaml_data):
#         if i.startswith("  CURRICULUM:"):
#             eval_yaml_data[idx] = f"  CURRICULUM: {args.curriculum}"
#         elif i.startswith("  MAX_EPISODE_STEPS:"):
#             eval_yaml_data[
#                 idx
#             ] = f"  MAX_EPISODE_STEPS: {int(args.max_num_steps * (1.0/args.robot_scale))}"
#         elif i.startswith("    RADIUS:"):
#             if args.robot_radius == -1.0:
#                 eval_yaml_data[idx] = f"    RADIUS: {robot_radius * args.robot_scale}"
#             else:
#                 eval_yaml_data[
#                     idx
#                 ] = f"    RADIUS: {args.robot_radius * args.robot_scale}"
#         elif i.startswith("    POSITION: [ 0.0, 1.5, 1.0 ]"):
#             position = (
#                 np.array([0.0, 0.35, 0.1])
#                 if args.robot_scale == 0.1
#                 else np.array([0.0, 1.5, 1.0])
#             )
#             eval_yaml_data[idx] = f"    POSITION: {position.tolist()}"
#         elif i.startswith(
#             "    POSITION: [ -0.03740343144695029, 0.5, -0.4164822634134684 ]"
#         ):
#             position = (
#                 np.array([-0.03740343144695029, 0.5, -0.4164822634134684])
#                 * args.robot_scale
#             )
#             eval_yaml_data[idx] = f"    POSITION: {list(position)}"
#         elif i.startswith(
#             "    POSITION: [ 0.03614789234067159, 0.5, -0.4164822634134684 ]"
#         ):
#             position = (
#                 np.array([0.03614789234067159, 0.5, -0.4164822634134684])
#                 * args.robot_scale
#             )
#             eval_yaml_data[idx] = f"    POSITION: {list(position)}"
#         elif i.startswith("    MAX_DEPTH:"):
#             max_depth = 3.5 * args.robot_scale
#             eval_yaml_data[idx] = f"    MAX_DEPTH: {max_depth}"
#         elif i.startswith("    ALLOW_SLIDING:"):
#             if args.sliding:
#                 eval_yaml_data[idx] = f"    ALLOW_SLIDING: True"
#         elif i.startswith("      CONTACT_TEST:"):
#             if args.no_contact_test:
#                 eval_yaml_data[idx] = f"      CONTACT_TEST: False"
#         elif i.startswith("  ROBOT:"):
#             eval_yaml_data[idx] = f"  ROBOT: '{robot}'"
#         elif i.startswith("    LOG_POINTGOAL:"):
#             if args.log_pointgoal:
#                 eval_yaml_data[idx] = f"    LOG_POINTGOAL: True"
#         elif i.startswith("    NORM_POINTGOAL:"):
#             if args.norm_pointgoal:
#                 eval_yaml_data[idx] = f"    NORM_POINTGOAL: True"
#         elif i.startswith("      ROBOT_URDF:"):
#             eval_yaml_data[idx] = f"      ROBOT_URDF: {robot_urdf}"
#         elif i.startswith("    MAP_RESOLUTION:"):
#             eval_yaml_data[idx] = f"    MAP_RESOLUTION: {args.map_resolution}"
#         elif i.startswith("    METERS_PER_PIXEL:"):
#             eval_yaml_data[
#                 idx
#             ] = f"    METERS_PER_PIXEL: {args.meters_per_pixel * args.robot_scale}"
#         elif i.startswith("    ROTATE_MAP:"):
#             if args.rotate_map:
#                 eval_yaml_data[idx] = f"    ROTATE_MAP: True"
#         elif i.startswith("    PAD_NOISE:"):
#             if args.pad_noise:
#                 eval_yaml_data[idx] = f"    PAD_NOISE: True"
#         elif i.startswith("    SECOND_CHANNEL:"):
#             if args.second_channel:
#                 eval_yaml_data[idx] = f"    SECOND_CHANNEL: True"
#         elif i.startswith("    MULTI_CHANNEL:"):
#             if args.multi_channel:
#                 eval_yaml_data[idx] = f"    MULTI_CHANNEL: True"
#         elif i.startswith("    DRAW_TRAJECTORY:"):
#             if args.draw_trajectory:
#                 eval_yaml_data[idx] = f"    DRAW_TRAJECTORY: True"
#         elif i.startswith("      NOISE_PERCENT:"):
#             eval_yaml_data[idx] = f"      NOISE_PERCENT: {args.context_sensor_noise}"
#         elif i.startswith("    DEBUG:"):
#             eval_yaml_data[idx] = f'    DEBUG: "{args.context_debug}"'
#         elif i.startswith("    USE_TOPDOWN_MAP:"):
#             if args.use_topdown_map:
#                 eval_yaml_data[idx] = f"    USE_TOPDOWN_MAP: True"
#         elif i.startswith("    CONTEXT_TYPE:"):
#             eval_yaml_data[idx] = f'    CONTEXT_TYPE: "{args.context_type}"'
#         elif i.startswith("  SENSORS:"):
#             pg = (
#                 "POINTGOAL_WITH_NOISY_GPS_COMPASS_SENSOR"
#                 if args.noisy_pointgoal
#                 else "POINTGOAL_WITH_GPS_COMPASS_SENSOR"
#             )
#             eval_yaml_data[idx] = f"  SENSORS: ['{pg}']"
#             if (
#                 args.context_map
#                 or args.context_resnet_map
#                 or args.context_waypoint
#                 or args.context_resnet_waypoint
#                 or args.context_map_trajectory
#                 or args.context_resnet_map_trajectory
#             ):
#                 if (args.context_map or args.context_resnet_map) and (
#                     args.context_waypoint or args.context_resnet_waypoint
#                 ):
#                     eval_yaml_data[
#                         idx
#                     ] = f"  SENSORS: ['{pg}', 'CONTEXT_MAP_SENSOR', 'CONTEXT_WAYPOINT_SENSOR']"
#                 else:
#                     if args.context_map or args.context_resnet_map:
#                         txt = "CONTEXT_MAP_SENSOR"
#                     elif args.context_waypoint or args.context_resnet_waypoint:
#                         txt = "CONTEXT_WAYPOINT_SENSOR"
#                     elif (
#                         args.context_map_trajectory
#                         or args.context_resnet_map_trajectory
#                     ):
#                         txt = "CONTEXT_MAP_TRAJECTORY_SENSOR"
#                     if args.no_pointgoal:
#                         eval_yaml_data[idx] = f"  SENSORS: ['{txt}']"
#                     else:
#                         eval_yaml_data[idx] = f"  SENSORS: ['{pg}', '{txt}']"
#         elif i.startswith("    PROJECT_GOAL:"):
#             eval_yaml_data[idx] = f"    PROJECT_GOAL: {args.project_goal}"
#         elif i.startswith("    BIN_POINTGOAL:"):
#             if args.target_encoding == "ans_bin":
#                 eval_yaml_data[idx] = f"    BIN_POINTGOAL: True"
#         elif i.startswith("  POSSIBLE_ACTIONS:"):
#             if args.control_type == "dynamic":
#                 control_type = "DYNAMIC_VELOCITY_CONTROL"
#             else:
#                 control_type = "VELOCITY_CONTROL"
#             eval_yaml_data[idx] = f'  POSSIBLE_ACTIONS: ["{control_type}"]'
#         elif i.startswith("    VELOCITY_CONTROL:"):
#             if args.control_type == "dynamic":
#                 eval_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
#         elif i.startswith("    DYNAMIC_VELOCITY_CONTROL:"):
#             if not args.control_type == "dynamic":
#                 eval_yaml_data[idx] = "    VELOCITY_CONTROL:"
#         elif i.startswith("      HOR_VEL_RANGE:"):
#             if args.velocity_y != -1.0:
#                 eval_yaml_data[
#                     idx
#                 ] = f"      HOR_VEL_RANGE: [ {-args.velocity_y, -args.velocity_y} ]"
#             if args.no_hor_vel:
#                 eval_yaml_data[idx] = "      HOR_VEL_RANGE: [ 0.0, 0.0 ]"
#         elif i.startswith("      LIN_VEL_RANGE:"):
#             lin_vel = 0.5 * args.robot_scale
#             eval_yaml_data[idx] = f"      LIN_VEL_RANGE: [ -{lin_vel}, {lin_vel} ]"
#         elif i.startswith("    POINTGOAL_SCALE:"):
#             eval_yaml_data[idx] = f"    POINTGOAL_SCALE: {1.0/args.robot_scale}"
#         elif i.startswith("    STACKED_MAP_RES:"):
#             if args.stacked_mpp:
#                 mpp_list = [float(i) for i in args.stacked_mpp]
#             else:
#                 mpp_list = [args.meters_per_pixel * args.robot_scale]
#             eval_yaml_data[idx] = f"    STACKED_MAP_RES: {mpp_list}"
#         elif i.startswith("      MIN_RAND_PITCH:"):
#             eval_yaml_data[idx] = f"      MIN_RAND_PITCH: {args.randomize_pitch_min}"
#         elif i.startswith("      MAX_RAND_PITCH:"):
#             eval_yaml_data[idx] = f"      MAX_RAND_PITCH: {args.randomize_pitch_max}"
#         elif i.startswith("      TIME_STEP:"):
#             eval_yaml_data[idx] = f"      TIME_STEP: {args.time_step}"
#             if args.control_type == "dynamic":
#                 eval_yaml_data[idx] = "      TIME_STEP: 0.33"
#         elif i.startswith("  SUCCESS_DISTANCE:"):
#             eval_yaml_data[idx] = f"  SUCCESS_DISTANCE: {robot_goal}"
#         elif i.startswith("    SUCCESS_DISTANCE:"):
#             eval_yaml_data[idx] = f"    SUCCESS_DISTANCE: {robot_goal}"
#         elif i.startswith("SEED:"):
#             eval_yaml_data[idx] = f"SEED: {args.seed}"
#         elif i.startswith("  DATA_PATH:"):
#             if args.dataset == "ferst":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/{split}/{split}.json.gz"
#             elif args.dataset == "coda":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test/coda_spot_test.json.gz"
#             elif args.dataset == "coda_straight_5m":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_5m/coda_spot_test_straight_5m.json.gz"
#             elif args.dataset == "coda_straight_6m":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_6m/coda_spot_test_straight_6m.json.gz"
#             elif args.dataset == "coda_straight_7m":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_7m/coda_spot_test_straight_7m.json.gz"
#             elif args.dataset == "coda_straight_8m":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_8m/coda_spot_test_straight_8m.json.gz"
#             elif args.dataset == "coda_straight_9m":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_9m/coda_spot_test_straight_9m.json.gz"
#             elif args.dataset == "coda_straight_10m":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_10m/coda_spot_test_straight_10m.json.gz"
#             elif args.dataset == "coda_straight_100m":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_100m/content/coda_spot_test_straight_100m.json.gz"
#             elif args.dataset == "coda_lobby":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_lobby/coda_lobby.json.gz"
#             elif args.dataset == "coda_lobby_hard":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_lobby_hard/coda_lobby_hard.json.gz"
#             elif args.dataset == "google":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/google/val/content/boulder4772-2_v2.json.gz"
#             elif args.dataset == "google_v3":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/google/val/content/boulder4772-2_v3.json.gz"
#             elif args.dataset == "google_val":
#                 data_path = (
#                     "/coc/testnvme/jtruong33/data/datasets/google/val_all/val.json.gz"
#                 )
#             elif args.dataset == "google_boulder":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/google/val_boulder/val_boulder.json.gz"
#             elif args.dataset == "ny":
#                 data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/{split}/{split}.json.gz"
#             elif args.dataset == "ny_train":
#                 data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/train/train.json.gz"
#             elif args.dataset == "hm3d_gibson_0.5":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson/pointnav_spot_0.5/{split}/{split}.json.gz"
#             elif args.dataset == "uf":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_unfurnished/{split}/{split}.json.gz"
#             elif args.dataset == "hm3d_uf":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d/pointnav_spot_unfurnished/{split}/{split}.json.gz"
#             elif args.dataset == "ferst_mf":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/multi_floor/{split}/{split}.json.gz"
#             elif args.dataset == "ferst_mf_1":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/multi_floor/val_1/val.json.gz"
#             elif args.dataset == "ferst_stairs":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/stairs/{split}/{split}.json.gz"
#             elif args.dataset == "hm3d_mf_only":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d/pointnav_spot_0.3_multi_floor_only/{split}/{split}.json.gz"
#             elif args.dataset == "hm3d_straight":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d/pointnav_spot_0.4_single_floor_long/val/content/HY1NcmCgn3n.json.gz"
#             elif args.dataset == "ny_google":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_ny_google/val/val.json.gz"
#             elif args.dataset == "google_1157":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/google/val_1157/content/mtv1157-1_lab.json.gz"
#             elif args.dataset == "ny_mini":
#                 data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson_ny/val_mini/val_mini.json.gz"
#             elif args.dataset == "blender":
#                 data_path = (
#                     "/coc/testnvme/jtruong33/data/datasets/blender/val/val.json.gz"
#                 )
#             eval_yaml_data[idx] = f"  DATA_PATH: {data_path}"
#         elif i.startswith("      noise_multiplier:"):
#             eval_yaml_data[idx] = f"      noise_multiplier: {args.noise_percent}"
#     with open(new_eval_task_yaml_path, "w") as f:
#         f.write("\n".join(eval_yaml_data))
#     print("Created " + new_eval_task_yaml_path)
#
#     # Edit the stored experiment yaml file
#     with open(exp_yaml_path) as f:
#         eval_exp_yaml_data = f.read().splitlines()
#
#     for idx, i in enumerate(eval_exp_yaml_data):
#         if i.startswith("BASE_TASK_CONFIG_PATH:"):
#             eval_exp_yaml_data[
#                 idx
#             ] = f"BASE_TASK_CONFIG_PATH: '{new_eval_task_yaml_path}'"
#         elif i.startswith("TENSORBOARD_DIR:"):
#             tb_dir = f"tb_eval_{args.control_type}"
#             if args.ckpt != -1:
#                 tb_dir += f"_ckpt_{args.ckpt}"
#             if args.video:
#                 tb_dir += "_video"
#             eval_exp_yaml_data[
#                 idx
#             ] = f"TENSORBOARD_DIR:    '{os.path.join(eval_dst_dir, 'tb_evals', tb_dir)}'"
#         elif i.startswith("NUM_PROCESSES:"):
#             eval_exp_yaml_data[idx] = "NUM_PROCESSES: 13"
#         elif i.startswith("    name:"):
#             if args.policy_name == "cnn":
#                 eval_exp_yaml_data[idx] = "    name: PointNavBaselinePolicy"
#             if args.outdoor_nav:
#                 eval_exp_yaml_data[idx] = "    name: OutdoorPolicy"
#             if args.context_map or args.context_waypoint or args.context_map_trajectory:
#                 eval_exp_yaml_data[idx] = "    name: PointNavContextPolicy"
#             if (
#                 args.context_resnet_map
#                 or args.context_resnet_waypoint
#                 or args.context_resnet_map_trajectory
#             ):
#                 eval_exp_yaml_data[idx] = "    name: PointNavResNetContextPolicy"
#             if args.rnn_type == "TRANSFORMER":
#                 eval_exp_yaml_data[idx] = "    name: PointNavContextSMTPolicy"
#         elif i.startswith("  COLLISION_PENALTY:"):
#             eval_exp_yaml_data[idx] = f"  COLLISION_PENALTY: {args.collision_penalty}"
#         elif i.startswith("CHECKPOINT_FOLDER:"):
#             eval_exp_yaml_data[
#                 idx
#             ] = f"CHECKPOINT_FOLDER:  '{os.path.join(dst_dir, 'checkpoints')}'"
#         elif i.startswith("EVAL_CKPT_PATH_DIR:"):
#             if args.ckpt == -1:
#                 eval_exp_yaml_data[
#                     idx
#                 ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}'"
#             else:
#                 eval_exp_yaml_data[
#                     idx
#                 ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}/ckpt.{args.ckpt}.pth'"
#         elif i.startswith("TXT_DIR:"):
#             txt_dir = f"txts_eval_{args.control_type}"
#             if args.ckpt != -1:
#                 txt_dir += f"_ckpt_{args.ckpt}"
#             eval_exp_yaml_data[
#                 idx
#             ] = f"TXT_DIR:            '{os.path.join(eval_dst_dir, 'txts', txt_dir)}'"
#         elif i.startswith("VIDEO_OPTION:"):
#             if args.video:
#                 eval_exp_yaml_data[idx] = "VIDEO_OPTION: ['disk']"
#             else:
#                 eval_exp_yaml_data[idx] = "VIDEO_OPTION: []"
#         elif i.startswith("NUM_ENVIRONMENTS:"):
#             eval_exp_yaml_data[idx] = f"NUM_ENVIRONMENTS: {args.num_environments}"
#             if (
#                 "ferst" in args.dataset
#                 or "coda" in args.dataset
#                 or args.dataset == "google"
#             ):
#                 eval_exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
#         elif i.startswith("SENSORS:"):
#             if args.video:
#                 eval_exp_yaml_data[
#                     idx
#                 ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR']"
#             if args.use_gray:
#                 eval_exp_yaml_data[
#                     idx
#                 ] = "SENSORS: ['SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
#                 if args.video:
#                     eval_exp_yaml_data[
#                         idx
#                     ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
#             elif args.use_gray_depth:
#                 eval_exp_yaml_data[
#                     idx
#                 ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
#                 if args.video:
#                     eval_exp_yaml_data[
#                         idx
#                     ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
#         elif i.startswith("VIDEO_DIR:"):
#             video_dir = (
#                 "video_dir"
#                 if args.ckpt == -1
#                 else f"video_dir_{args.control_type}_ckpt_{args.ckpt}"
#             )
#             eval_exp_yaml_data[
#                 idx
#             ] = f"VIDEO_DIR:          '{os.path.join(eval_dst_dir, 'videos', video_dir)}'"
#         elif i.startswith("    SPLIT:"):
#             if args.dataset == "hm3d":
#                 eval_exp_yaml_data[idx] = "    SPLIT: val"
#             elif args.dataset == "ferst_20m":
#                 eval_exp_yaml_data[idx] = "    SPLIT: val_20m"
#         elif i.startswith("    in_channels:"):
#             num_mpp = len(mpp_list)
#             if args.second_channel:
#                 num_mpp += 1
#             eval_exp_yaml_data[idx] = f"    in_channels: {num_mpp}"
#         elif i.startswith("    action_distribution_type:"):
#             if args.discrete:
#                 eval_exp_yaml_data[idx] = "    action_distribution_type: 'categorical'"
#         elif i.startswith("      use_log_std:"):
#             if args.log_std:
#                 eval_exp_yaml_data[idx] = "      use_log_std: True"
#         elif i.startswith("    tgt_encoding:"):
#             eval_exp_yaml_data[idx] = f"    tgt_encoding: '{args.target_encoding}'"
#         elif i.startswith("    use_waypoint_encoder:"):
#             if args.use_waypoint_encoder:
#                 eval_exp_yaml_data[idx] = f"    use_waypoint_encoder: True"
#         elif i.startswith("    num_cnns:"):
#             if args.two_cnns:
#                 eval_exp_yaml_data[idx] = "    num_cnns: 2"
#         elif i.startswith("      ENABLED_TRANSFORMS: [ ]"):
#             if args.pepper_noise:
#                 eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['PEPPER_NOISE']"
#             elif args.cutout_noise:
#                 eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['CUTOUT']"
#             elif args.median_blur:
#                 eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['MEDIAN_BLUR']"
#         elif i.startswith("        NOISE_PERCENT:"):
#             eval_exp_yaml_data[idx] = f"        NOISE_PERCENT: {args.noise_percent}"
#         elif i.startswith("        KERNEL_SIZE:"):
#             eval_exp_yaml_data[idx] = f"        KERNEL_SIZE: {args.kernel_size}"
#         elif i.startswith("    context_hidden_size:"):
#             eval_exp_yaml_data[
#                 idx
#             ] = f"    context_hidden_size: {args.context_hidden_size}"
#         elif i.startswith("    tgt_hidden_size:"):
#             eval_exp_yaml_data[idx] = f"    tgt_hidden_size: {args.tgt_hidden_size}"
#         elif i.startswith("    use_prev_action:"):
#             if args.use_prev_action:
#                 eval_exp_yaml_data[idx] = f"    use_prev_action: True"
#         elif i.startswith("    cnn_type:"):
#             eval_exp_yaml_data[idx] = f"    cnn_type: '{args.cnn_type}'"
#         elif i.startswith("    decoder_output:"):
#             if args.splitnet and args.surface_normal:
#                 eval_exp_yaml_data[
#                     idx
#                 ] = "    decoder_output: ['depth', 'surface_normals']"
#         elif i.startswith("    visual_encoder:"):
#             if args.splitnet:
#                 if args.visual_encoder == "shallow":
#                     visual_encoder = "ShallowVisualEncoder"
#                 elif args.visual_encoder == "resnet":
#                     visual_encoder = "BaseResNetEncoder"
#                 eval_exp_yaml_data[idx] = f"    visual_encoder: {visual_encoder}"
#         elif i.startswith("    pretrained_weights: "):
#             if args.pretrained_weights != "":
#                 eval_exp_yaml_data[
#                     idx
#                 ] = f"    pretrained_weights: {args.pretrained_weights}"
#         elif i.startswith("    pretrained_encoder: "):
#             if args.pretrained_encoder:
#                 eval_exp_yaml_data[idx] = f"    pretrained_encoder: True"
#         elif i.startswith("    rnn_type:"):
#             eval_exp_yaml_data[idx] = f"    rnn_type: {args.rnn_type}"
#         elif i.startswith("    num_recurrent_layers:"):
#             eval_exp_yaml_data[
#                 idx
#             ] = f"    num_recurrent_layers: {args.num_recurrent_layers}"
#         elif i.startswith("      nhead:"):
#             eval_exp_yaml_data[idx] = f"      nhead: {args.nhead}"
#         elif i.startswith("      num_encoder_layers:"):
#             eval_exp_yaml_data[
#                 idx
#             ] = f"      num_encoder_layers: {args.num_encoder_layers}"
#
#     if os.path.isdir(tb_dir):
#         response = input(
#             f"{tb_dir} directory already exists. Delete, continue, or abort? [d/c/A]: "
#         )
#         if response == "d":
#             print(f"Deleting {tb_dir}")
#             shutil.rmtree(tb_dir)
#         elif response == "c":
#             print("Continuing.")
#         else:
#             print("Aborting.")
#             exit()
#
#     with open(new_eval_exp_yaml_path, "w") as f:
#         f.write("\n".join(eval_exp_yaml_data))
#     print("Created " + new_eval_exp_yaml_path)
#
#     eval_experiment_name = experiment_name + exp_name
#
#     # Create slurm job
#     with open(EVAL_SLURM_TEMPLATE) as f:
#         slurm_data = f.read()
#         slurm_data = slurm_data.replace("$TEMPLATE", eval_experiment_name)
#         slurm_data = slurm_data.replace("$CONDA_ENV", CONDA_ENV)
#         slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", HABITAT_LAB)
#         slurm_data = slurm_data.replace(
#             "$LOG", os.path.join(eval_dst_dir, eval_experiment_name)
#         )
#         slurm_data = slurm_data.replace("$CONFIG_YAML", new_eval_exp_yaml_path)
#         slurm_data = slurm_data.replace("$PARTITION", args.partition)
#         if args.partition == "overcap":
#             slurm_data = slurm_data.replace("# ACCOUNT", "#SBATCH --account overcap")
#     slurm_path = os.path.join(eval_dst_dir, eval_experiment_name + ".sh")
#     with open(slurm_path, "w") as f:
#         f.write(slurm_data)
#     print("Generated slurm job: " + slurm_path)
#
#     if not args.x:
#         # Submit slurm job
#         cmd = "sbatch " + slurm_path
#         subprocess.check_call(cmd.split(), cwd=dst_dir)
#     else:
#         print(slurm_data)
#     print(
#         f"\nSee output with:\ntail -F {os.path.join(eval_dst_dir, eval_experiment_name + '.err')}"
#     )
