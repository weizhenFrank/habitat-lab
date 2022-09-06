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

automate_command = "python " + " ".join(sys.argv)
HABITAT_LAB = "/coc/testnvme/jtruong33/google_nav/habitat-lab"
CONDA_ENV = "/nethome/jtruong33/miniconda3/envs/habitat-outdoor/bin/python"
RESULTS = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results"
# RESULTS = "/coc/testnvme/jtruong33/results/outdoor_nav_results"
SLURM_TEMPLATE = os.path.join(HABITAT_LAB, "slurm_template.sh")
EVAL_SLURM_TEMPLATE = os.path.join(HABITAT_LAB, "eval_slurm_template.sh")

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")

# Training
parser.add_argument("-sd", "--seed", type=int, default=1)
parser.add_argument("-ns", "--max_num_steps", type=int, default=500)
parser.add_argument("-r", "--robot", default="Spot")
parser.add_argument("-rr", "--robot-radius", type=float, default=-1.0)
parser.add_argument("-c", "--control-type", default="kinematic")
parser.add_argument("-p", "--partition", default="long")
parser.add_argument("-s", "--sliding", default=False, action="store_true")
parser.add_argument("-nct", "--no-contact-test", default=False, action="store_true")
parser.add_argument("-nhv", "--no-hor-vel", default=False, action="store_true")
parser.add_argument("-cp", "--collision-penalty", type=float, default=0.003)
parser.add_argument("-bp", "--backwards-penalty", type=float, default=0.03)
parser.add_argument("-ap", "--acc-penalty", type=float, default=0.0)
parser.add_argument("-vy", "--velocity-y", type=float, default=-1.0)
parser.add_argument("-rpl", "--randomize-pitch-min", type=float, default=0.0)
parser.add_argument("-rpu", "--randomize-pitch-max", type=float, default=0.0)
parser.add_argument("-pg", "--project-goal", type=float, default=-1.0)
parser.add_argument("-ts", "--time-step", type=float, default=1.0)
parser.add_argument("-odn", "--outdoor-nav", default=False, action="store_true")
parser.add_argument("-cm", "--context-map", default=False, action="store_true")
parser.add_argument("-cw", "--context-waypoint", default=False, action="store_true")
parser.add_argument(
    "-cmt", "--context-map-trajectory", default=False, action="store_true"
)
parser.add_argument(
    "-wpte", "--use-waypoint-encoder", default=False, action="store_true"
)
parser.add_argument("-crm", "--context-resnet-map", default=False, action="store_true")
parser.add_argument(
    "-crw", "--context-resnet-waypoint", default=False, action="store_true"
)
parser.add_argument(
    "-crmt", "--context-resnet-map-trajectory", default=False, action="store_true"
)
parser.add_argument("-sc", "--second-channel", default=False, action="store_true")
parser.add_argument("-mc", "--multi-channel", default=False, action="store_true")
parser.add_argument("-csn", "--context-sensor-noise", type=float, default=0.0)
parser.add_argument("-chs", "--context-hidden-size", type=int, default=512)
parser.add_argument("-ths", "--tgt-hidden-size", type=int, default=512)
parser.add_argument("-cd", "--context-debug", default="")
parser.add_argument("-ct", "--context-type", default="MAP")
parser.add_argument("-mr", "--map-resolution", type=int, default=100)
parser.add_argument("-mpp", "--meters-per-pixel", type=float, default=0.5)
parser.add_argument("-rotm", "--rotate-map", default=False, action="store_true")
parser.add_argument("-lpg", "--log-pointgoal", default=False, action="store_true")
parser.add_argument("-lstd", "--log-std", default=False, action="store_true")
parser.add_argument("-pa", "--use-prev-action", default=False, action="store_true")
parser.add_argument("-cnnt", "--cnn-type", default="cnn_2d")
parser.add_argument("-rnnt", "--rnn-type", default="GRU")
parser.add_argument("-nrl", "--num_recurrent_layers", type=int, default=1)
parser.add_argument("-tgte", "--target_encoding", default="linear_2")
parser.add_argument("-ft", "--finetune", default=False, action="store_true")
parser.add_argument("-npg", "--noisy-pointgoal", default=False, action="store_true")
parser.add_argument("--constraint", default="x")

## options for dataset are hm3d_gibson, hm3d, gibson
parser.add_argument("--policy-name", default="PointNavResNetPolicy")
parser.add_argument("-ds", "--dataset", default="hm3d_gibson")
parser.add_argument("-ne", "--num_environments", type=int, default=24)
parser.add_argument("-ngpu", "--num_gpus", type=int, default=8)

parser.add_argument("-g", "--use_gray", default=False, action="store_true")
parser.add_argument("-gd", "--use_gray_depth", default=False, action="store_true")
parser.add_argument("--coda", default=False, action="store_true")

parser.add_argument("--splitnet", default=False, action="store_true")
parser.add_argument("-sn", "--surface_normal", default=False, action="store_true")
parser.add_argument("-ve", "--visual-encoder", default="resnet")
parser.add_argument("-ml", "--motion_loss", default=False, action="store_true")

parser.add_argument("-2cnn", "--two_cnns", default=False, action="store_true")

parser.add_argument("-pn", "--pepper_noise", default=False, action="store_true")
parser.add_argument("-rn", "--redwood_noise", default=False, action="store_true")
parser.add_argument("-mb", "--median_blur", default=False, action="store_true")
parser.add_argument("-ks", "--kernel_size", type=float, default=9)
parser.add_argument("-np", "--noise_percent", type=float, default=0.4)
parser.add_argument("-cn", "--cutout_noise", default=False, action="store_true")
parser.add_argument("-curr", "--curriculum", default=False, action="store_true")

# Evaluation
parser.add_argument("-e", "--eval", default=False, action="store_true")
parser.add_argument("-cpt", "--ckpt", type=int, default=-1)
parser.add_argument("-v", "--video", default=False, action="store_true")

parser.add_argument("-d", "--debug", default=False, action="store_true")
parser.add_argument("-x", default=False, action="store_true")
parser.add_argument("--ext", default="")
args = parser.parse_args()

if args.splitnet:
    EXP_YAML = "habitat_baselines/config/pointnav/ddppo_pointnav_spot_splitnet.yaml"
else:
    EXP_YAML = "habitat_baselines/config/pointnav/ddppo_pointnav_spot.yaml"

if args.redwood_noise:
    TASK_YAML = "configs/tasks/pointnav_spot_redwood.yaml"
else:
    TASK_YAML = "configs/tasks/pointnav_spot.yaml"

experiment_name = args.experiment_name

dst_dir = os.path.join(RESULTS, experiment_name)
eval_dst_dir = os.path.join(RESULTS, experiment_name, "eval", args.control_type)

exp_yaml_path = os.path.join(HABITAT_LAB, EXP_YAML)
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
if args.pepper_noise:
    exp_name += f"_pepper_noise_{args.noise_percent}"
    eval_dst_dir += f"_pepper_noise_{args.noise_percent}"
if args.redwood_noise:
    exp_name += f"_redwood_noise_{args.noise_percent}"
    eval_dst_dir += f"_redwood_noise_{args.noise_percent}"
if args.median_blur:
    exp_name += f"_median_blur_{args.kernel_size}"
    eval_dst_dir += f"_median_blur_{args.kernel_size}"
if args.sliding:
    eval_dst_dir += f"_sliding"
if args.no_contact_test:
    eval_dst_dir += f"_no_contact_test"

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

robot_urdfs_dict = {
    "A1": "/coc/testnvme/jtruong33/data/URDF_demo_assets/a1/a1.urdf",
    "AlienGo": "/coc/testnvme/jtruong33/data/URDF_demo_assets/aliengo/urdf/aliengo.urdf",
    "Daisy": "/coc/testnvme/jtruong33/data/URDF_demo_assets/daisy/daisy_advanced_akshara.urdf",
    "Spot": "/coc/testnvme/jtruong33/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid_rot_fix.urdf",
    "Locobot": "/coc/testnvme/jtruong33/data/URDF_demo_assets/locobot/urdf/locobot_description2.urdf",
}

robot_goal_dict = {
    "A1": 0.24,
    "AlienGo": 0.3235,
    "Locobot": 0.20,
    "Spot": 0.425,
}
robot_radius_dict = {"A1": 0.2, "AlienGo": 0.22, "Locobot": 0.23, "Spot": 0.3}

robot = args.robot
robot_goal = robot_goal_dict[robot]
robot_urdf = robot_urdfs_dict[robot]

succ_radius = robot_goal_dict[robot]
robot_radius = robot_radius_dict[robot]

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
        if i.startswith("  CURRICULUM:"):
            task_yaml_data[idx] = f"  CURRICULUM: {args.curriculum}"
        elif i.startswith("  MAX_EPISODE_STEPS:"):
            task_yaml_data[idx] = f"  MAX_EPISODE_STEPS: {args.max_num_steps}"
        elif i.startswith("    RADIUS:"):
            if args.robot_radius == -1.0:
                task_yaml_data[idx] = f"    RADIUS: {robot_radius}"
            else:
                task_yaml_data[idx] = f"    RADIUS: {args.robot_radius}"
        elif i.startswith("  ROBOT:"):
            task_yaml_data[idx] = f"  ROBOT: '{robot}'"
        elif i.startswith("  SENSORS:"):
            pg = (
                "POINTGOAL_WITH_NOISY_GPS_COMPASS_SENSOR"
                if args.noisy_pointgoal
                else "POINTGOAL_WITH_GPS_COMPASS_SENSOR"
            )
            task_yaml_data[idx] = f"  SENSORS: ['{pg}']"
            if (args.context_map or args.context_resnet_map) and (
                args.context_waypoint or args.context_resnet_waypoint
            ):
                task_yaml_data[
                    idx
                ] = f"  SENSORS: ['{pg}', 'CONTEXT_MAP_SENSOR', 'CONTEXT_WAYPOINT_SENSOR']"
            elif args.context_map or args.context_resnet_map:
                task_yaml_data[idx] = f"  SENSORS: ['{pg}', 'CONTEXT_MAP_SENSOR']"
            elif args.context_waypoint or args.context_resnet_waypoint:
                task_yaml_data[idx] = f"  SENSORS: ['{pg}', 'CONTEXT_WAYPOINT_SENSOR']"
            elif args.context_map_trajectory or args.context_resnet_map_trajectory:
                task_yaml_data[
                    idx
                ] = f"  SENSORS: ['{pg}', 'CONTEXT_MAP_TRAJECTORY_SENSOR']"
        elif i.startswith("    PROJECT_GOAL:"):
            task_yaml_data[idx] = f"    PROJECT_GOAL: {args.project_goal}"
        elif i.startswith("    BIN_POINTGOAL:"):
            if args.target_encoding == "ans_bin":
                task_yaml_data[idx] = f"    BIN_POINTGOAL: True"
        elif i.startswith("      ROBOT_URDF:"):
            task_yaml_data[idx] = f"      ROBOT_URDF: {robot_urdf}"
        elif i.startswith("    MAP_RESOLUTION:"):
            task_yaml_data[idx] = f"    MAP_RESOLUTION: {args.map_resolution}"
        elif i.startswith("    METERS_PER_PIXEL:"):
            task_yaml_data[idx] = f"    METERS_PER_PIXEL: {args.meters_per_pixel}"
        elif i.startswith("    ROTATE_MAP:"):
            if args.rotate_map:
                task_yaml_data[idx] = f"    ROTATE_MAP: True"
        elif i.startswith("    SECOND_CHANNEL:"):
            if args.second_channel:
                task_yaml_data[idx] = f"    SECOND_CHANNEL: True"
        elif i.startswith("    MULTI_CHANNEL:"):
            if args.multi_channel:
                task_yaml_data[idx] = f"    MULTI_CHANNEL: True"
        elif i.startswith("      NOISE_PERCENT:"):
            task_yaml_data[idx] = f"      NOISE_PERCENT: {args.context_sensor_noise}"
        elif i.startswith("    DEBUG:"):
            task_yaml_data[idx] = f'    DEBUG: "{args.context_debug}"'
        elif i.startswith("    CONTEXT_TYPE:"):
            task_yaml_data[idx] = f'    CONTEXT_TYPE: "{args.context_type}"'
        elif i.startswith("  POSSIBLE_ACTIONS:"):
            if args.control_type == "dynamic":
                control_type = "DYNAMIC_VELOCITY_CONTROL"
            else:
                control_type = "VELOCITY_CONTROL"
            task_yaml_data[idx] = f'  POSSIBLE_ACTIONS: ["{control_type}"]'
        elif i.startswith("    VELOCITY_CONTROL:"):
            if args.control_type == "dynamic":
                task_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
        elif i.startswith("    DYNAMIC_VELOCITY_CONTROL:"):
            if not args.control_type == "dynamic":
                task_yaml_data[idx] = "    VELOCITY_CONTROL:"
        elif i.startswith("      HOR_VEL_RANGE:"):
            if args.velocity_y != -1.0:
                task_yaml_data[
                    idx
                ] = f"      HOR_VEL_RANGE: [ {-args.velocity_y, -args.velocity_y} ]"
            if args.no_hor_vel:
                task_yaml_data[idx] = "      HOR_VEL_RANGE: [ 0.0, 0.0 ]"
        elif i.startswith("      MIN_RAND_PITCH:"):
            task_yaml_data[idx] = f"      MIN_RAND_PITCH: {args.randomize_pitch_min}"
        elif i.startswith("      MAX_RAND_PITCH:"):
            task_yaml_data[idx] = f"      MAX_RAND_PITCH: {args.randomize_pitch_max}"
        elif i.startswith("      TIME_STEP:"):
            task_yaml_data[idx] = f"      TIME_STEP: {args.time_step}"
            if args.control_type == "dynamic":
                task_yaml_data[idx] = "      TIME_STEP: 0.33"
        elif i.startswith("  SUCCESS_DISTANCE:"):
            task_yaml_data[idx] = f"  SUCCESS_DISTANCE: {succ_radius - 0.05}"
        elif i.startswith("    SUCCESS_DISTANCE:"):
            task_yaml_data[idx] = f"    SUCCESS_DISTANCE: {succ_radius - 0.05}"
        elif i.startswith("SEED:"):
            task_yaml_data[idx] = f"SEED: {args.seed}"
        elif i.startswith("  DATA_PATH:"):
            if args.dataset == "ferst":
                data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test/coda_spot_test.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda_5m_straight":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_5m/coda_spot_test_straight_5m.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda_6m_straight":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_6m/coda_spot_test_straight_6m.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda_7m_straight":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_7m/coda_spot_test_straight_7m.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda_8m_straight":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_8m/coda_spot_test_straight_8m.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda_9m_straight":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_9m/coda_spot_test_straight_9m.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda_10m_straight":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_10m/coda_spot_test_straight_10m.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda_lobby":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_lobby/coda_lobby.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "coda_lobby_hard":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_lobby_hard/coda_lobby_hard.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "ny":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "hm3d_gibson_0.5":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson/pointnav_spot_0.5/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "ny_uf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson_ny_unfurnished/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "uf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_unfurnished/train/content/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "hm3d_mf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d/pointnav_spot_0.3_multi_floor/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "hm3d_mf_long":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_unfurnished_hm3d_mf_long/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "hm3d_mf_uf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_mf_unfurnished/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "uf_mf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_unfurnished_hm3d_mf/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "uf_hm3d_0.4_mf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_unfurnished_0.4_multi_floor_long/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
            elif args.dataset == "uf_hm3d_0.4_sf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_unfurnished_0.4_single_floor_long/{split}/{split}.json.gz"
                task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
        elif i.startswith("      noise_multiplier:"):
            task_yaml_data[idx] = f"      noise_multiplier: {args.noise_percent}"
    with open(new_task_yaml_path, "w") as f:
        f.write("\n".join(task_yaml_data))
    print("Created " + new_task_yaml_path)

    # Create experiment yaml file, using file within Habitat Lab repo as a template
    with open(exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(exp_yaml_data):
        if i.startswith("BASE_TASK_CONFIG_PATH:"):
            exp_yaml_data[idx] = f"BASE_TASK_CONFIG_PATH: '{new_task_yaml_path}'"
        elif i.startswith("TOTAL_NUM_STEPS:"):
            max_num_steps = 5e8 if args.control_type == "kinematic" else 5e7
            exp_yaml_data[idx] = f"TOTAL_NUM_STEPS: {max_num_steps}"
        elif i.startswith("TENSORBOARD_DIR:"):
            exp_yaml_data[idx] = f"TENSORBOARD_DIR:    '{os.path.join(dst_dir, 'tb')}'"
        elif i.startswith("NUM_ENVIRONMENTS:"):
            exp_yaml_data[idx] = f"NUM_ENVIRONMENTS: {args.num_environments}"
            if "ferst" in args.dataset or "coda" in args.dataset:
                exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
        elif i.startswith("  COLLISION_PENALTY:"):
            exp_yaml_data[idx] = f"  COLLISION_PENALTY: {args.collision_penalty}"
        elif i.startswith("  BACKWARDS_PENALTY:"):
            exp_yaml_data[idx] = f"  BACKWARDS_PENALTY: {args.backwards_penalty}"
        elif i.startswith("  ANG_ACCEL_PENALTY_COEFF:"):
            exp_yaml_data[idx] = f"  ANG_ACCEL_PENALTY_COEFF: {args.acc_penalty}"
        elif i.startswith("SENSORS:"):
            if args.use_gray:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
            elif args.use_gray_depth:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
            if args.surface_normal:
                exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_SURFACE_NORMAL_SENSOR']"
        elif i.startswith("VIDEO_DIR:"):
            exp_yaml_data[
                idx
            ] = f"VIDEO_DIR:          '{os.path.join(dst_dir, 'video_dir')}'"
        elif i.startswith("EVAL_CKPT_PATH_DIR:"):
            exp_yaml_data[
                idx
            ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("CHECKPOINT_FOLDER:"):
            exp_yaml_data[
                idx
            ] = f"CHECKPOINT_FOLDER:  '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("TXT_DIR:"):
            exp_yaml_data[
                idx
            ] = f"TXT_DIR:            '{os.path.join(dst_dir, 'txts')}'"
        elif i.startswith("      use_log_std:"):
            if args.log_std:
                exp_yaml_data[idx] = "      use_log_std: True"
        elif i.startswith("    tgt_encoding:"):
            exp_yaml_data[idx] = f"    tgt_encoding: '{args.target_encoding}'"
        elif i.startswith("    use_waypoint_encoder:"):
            if args.use_waypoint_encoder:
                exp_yaml_data[idx] = f"    use_waypoint_encoder: True"
        elif i.startswith("    context_hidden_size:"):
            exp_yaml_data[idx] = f"    context_hidden_size: {args.context_hidden_size}"
        elif i.startswith("    tgt_hidden_size:"):
            exp_yaml_data[idx] = f"    tgt_hidden_size: {args.tgt_hidden_size}"
        elif i.startswith("    use_prev_action:"):
            if args.use_prev_action:
                exp_yaml_data[idx] = f"    use_prev_action: True"
        elif i.startswith("    cnn_type:"):
            exp_yaml_data[idx] = f"    cnn_type: '{args.cnn_type}'"
        elif i.startswith("    num_cnns:"):
            if args.two_cnns:
                exp_yaml_data[idx] = "    num_cnns: 2"
        elif i.startswith("    name:"):
            if args.policy_name == "cnn":
                exp_yaml_data[idx] = "    name: PointNavBaselinePolicy"
            if args.outdoor_nav:
                exp_yaml_data[idx] = "    name: OutdoorPolicy"
            if args.context_map or args.context_waypoint or args.context_map_trajectory:
                exp_yaml_data[idx] = "    name: PointNavContextPolicy"
            if args.context_resnet_map or args.context_resnet_waypoint:
                exp_yaml_data[idx] = "    name: PointNavResNetContextPolicy"
            if args.rnn_type == "TRANSFORMER":
                exp_yaml_data[idx] = "    name: PointNavContextSMTPolicy"
        elif i.startswith("      ENABLED_TRANSFORMS: [ ]"):
            if args.pepper_noise:
                exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['PEPPER_NOISE']"
            elif args.cutout_noise:
                exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['CUTOUT']"
            elif args.median_blur:
                exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['MEDIAN_BLUR']"
        elif i.startswith("        NOISE_PERCENT:"):
            exp_yaml_data[idx] = f"        NOISE_PERCENT: {args.noise_percent}"
        elif i.startswith("        KERNEL_SIZE:"):
            exp_yaml_data[idx] = f"        KERNEL_SIZE: {args.kernel_size}"
        elif i.startswith("    decoder_output:"):
            if args.splitnet and args.surface_normal:
                exp_yaml_data[idx] = "    decoder_output: ['depth', 'surface_normals']"
        elif i.startswith("    use_motion_loss:"):
            if args.splitnet and args.motion_loss:
                exp_yaml_data[idx] = "    use_visual_loss: True"
        elif i.startswith("    update_motion_decoder_features:"):
            if args.splitnet and args.motion_loss:
                exp_yaml_data[idx] = "    update_motion_decoder_features: True"
        elif i.startswith("    visual_encoder:"):
            if args.splitnet:
                if args.visual_encoder == "shallow":
                    visual_encoder = "ShallowVisualEncoder"
                elif args.visual_encoder == "resnet":
                    visual_encoder = "BaseResNetEncoder"
                exp_yaml_data[idx] = f"    visual_encoder: {visual_encoder}"
        elif i.startswith("    tasks: "):
            if args.splitnet:
                if args.motion_loss:
                    exp_yaml_data[
                        idx
                    ] = f"    tasks: ['VisualReconstructionTask', 'EgomotionPredictionTask']"
                else:
                    exp_yaml_data[idx] = f"    tasks: ['VisualReconstructionTask']"
        elif i.startswith("  USE_OUTDOOR:"):
            if args.outdoor_nav:
                exp_yaml_data[idx] = f"  USE_OUTDOOR: True"
        elif i.startswith("    pretrained_weights: "):
            if args.finetune:
                # ft_weights = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_simple_cnn_cutout_nhy_2hz_ny_rand_pitch_bp_0.03_sd_1/checkpoints/ckpt.84.pth"
                # ft_weights = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_simple_cnn_cutout_nhy_2hz_hm3d_mf_rand_pitch_-1.0_1.0_bp_0.03_log_sd_1/checkpoints/ckpt.95.pth"
                ft_weights = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_simple_cnn_cutout_nhy_2hz_ny_rand_pitch_-1.0_1.0_bp_0.03_sd_1_16env_context_no_noise_log_agent_rot/checkpoints/ckpt.94.pth"
                exp_yaml_data[idx] = f"    pretrained_weights: {ft_weights}"
        elif i.startswith("    pretrained: "):
            if args.finetune:
                exp_yaml_data[idx] = f"    pretrained: True"
        elif i.startswith("    rnn_type:"):
            exp_yaml_data[idx] = f"    rnn_type: {args.rnn_type}"
        elif i.startswith("    num_recurrent_layers:"):
            exp_yaml_data[
                idx
            ] = f"    num_recurrent_layers: {args.num_recurrent_layers}"
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

# Evaluation
else:
    # Make sure folder exists
    assert os.path.isdir(dst_dir), f"{dst_dir} directory does not exist"
    os.makedirs(eval_dst_dir, exist_ok=True)
    with open(os.path.join(eval_dst_dir, "automate_job_cmd.txt"), "w") as f:
        f.write(automate_command)
    print(
        "Saved automate command: " + os.path.join(eval_dst_dir, "automate_job_cmd.txt")
    )

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        eval_yaml_data = f.read().splitlines()

    for idx, i in enumerate(eval_yaml_data):
        if i.startswith("  CURRICULUM:"):
            eval_yaml_data[idx] = f"  CURRICULUM: {args.curriculum}"
        elif i.startswith("  MAX_EPISODE_STEPS:"):
            eval_yaml_data[idx] = f"  MAX_EPISODE_STEPS: {args.max_num_steps}"
        elif i.startswith("    RADIUS:"):
            if args.robot_radius == -1.0:
                eval_yaml_data[idx] = f"    RADIUS: {robot_radius}"
            else:
                eval_yaml_data[idx] = f"    RADIUS: {args.robot_radius}"
        elif i.startswith("    ALLOW_SLIDING:"):
            if args.sliding:
                eval_yaml_data[idx] = f"    ALLOW_SLIDING: True"
        elif i.startswith("      CONTACT_TEST:"):
            if args.no_contact_test:
                eval_yaml_data[idx] = f"      CONTACT_TEST: False"
        elif i.startswith("  ROBOT:"):
            eval_yaml_data[idx] = f"  ROBOT: '{robot}'"
        elif i.startswith("      ROBOT_URDF:"):
            eval_yaml_data[idx] = f"      ROBOT_URDF: {robot_urdf}"
        elif i.startswith("    MAP_RESOLUTION:"):
            eval_yaml_data[idx] = f"    MAP_RESOLUTION: {args.map_resolution}"
        elif i.startswith("    METERS_PER_PIXEL:"):
            eval_yaml_data[idx] = f"    METERS_PER_PIXEL: {args.meters_per_pixel}"
        elif i.startswith("    ROTATE_MAP:"):
            if args.rotate_map:
                eval_yaml_data[idx] = f"    ROTATE_MAP: True"
        elif i.startswith("    SECOND_CHANNEL:"):
            if args.second_channel:
                eval_yaml_data[idx] = f"    SECOND_CHANNEL: True"
        elif i.startswith("    MULTI_CHANNEL:"):
            if args.multi_channel:
                eval_yaml_data[idx] = f"    MULTI_CHANNEL: True"
        elif i.startswith("      NOISE_PERCENT:"):
            eval_yaml_data[idx] = f"      NOISE_PERCENT: {args.context_sensor_noise}"
        elif i.startswith("    DEBUG:"):
            eval_yaml_data[idx] = f'    DEBUG: "{args.context_debug}"'
        elif i.startswith("    CONTEXT_TYPE:"):
            eval_yaml_data[idx] = f'    CONTEXT_TYPE: "{args.context_type}"'
        elif i.startswith("  SENSORS:"):
            pg = (
                "POINTGOAL_WITH_NOISY_GPS_COMPASS_SENSOR"
                if args.noisy_pointgoal
                else "POINTGOAL_WITH_GPS_COMPASS_SENSOR"
            )
            eval_yaml_data[idx] = f"  SENSORS: ['{pg}']"
            if (args.context_map or args.context_resnet_map) and (
                args.context_waypoint or args.context_resnet_waypoint
            ):
                eval_yaml_data[
                    idx
                ] = f"  SENSORS: ['{pg}', 'CONTEXT_MAP_SENSOR', 'CONTEXT_WAYPOINT_SENSOR']"
            elif args.context_map or args.context_resnet_map:
                eval_yaml_data[idx] = f"  SENSORS: ['{pg}', 'CONTEXT_MAP_SENSOR']"
            elif args.context_waypoint or args.context_resnet_waypoint:
                eval_yaml_data[idx] = f"  SENSORS: ['{pg}', 'CONTEXT_WAYPOINT_SENSOR']"
            elif args.context_map_trajectory or args.context_resnet_map_trajectory:
                eval_yaml_data[
                    idx
                ] = f"  SENSORS: ['{pg}', 'CONTEXT_MAP_TRAJECTORY_SENSOR']"
        elif i.startswith("    PROJECT_GOAL:"):
            eval_yaml_data[idx] = f"    PROJECT_GOAL: {args.project_goal}"
        elif i.startswith("    BIN_POINTGOAL:"):
            if args.target_encoding == "ans_bin":
                eval_yaml_data[idx] = f"    BIN_POINTGOAL: True"
        elif i.startswith("  POSSIBLE_ACTIONS:"):
            if args.control_type == "dynamic":
                control_type = "DYNAMIC_VELOCITY_CONTROL"
            else:
                control_type = "VELOCITY_CONTROL"
            eval_yaml_data[idx] = f'  POSSIBLE_ACTIONS: ["{control_type}"]'
        elif i.startswith("    VELOCITY_CONTROL:"):
            if args.control_type == "dynamic":
                eval_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
        elif i.startswith("    DYNAMIC_VELOCITY_CONTROL:"):
            if not args.control_type == "dynamic":
                eval_yaml_data[idx] = "    VELOCITY_CONTROL:"
        elif i.startswith("      HOR_VEL_RANGE:"):
            if args.velocity_y != -1.0:
                eval_yaml_data[
                    idx
                ] = f"      HOR_VEL_RANGE: [ {-args.velocity_y, -args.velocity_y} ]"
            if args.no_hor_vel:
                eval_yaml_data[idx] = "      HOR_VEL_RANGE: [ 0.0, 0.0 ]"
        elif i.startswith("      MIN_RAND_PITCH:"):
            eval_yaml_data[idx] = f"      MIN_RAND_PITCH: {args.randomize_pitch_min}"
        elif i.startswith("      MAX_RAND_PITCH:"):
            eval_yaml_data[idx] = f"      MAX_RAND_PITCH: {args.randomize_pitch_max}"
        elif i.startswith("      TIME_STEP:"):
            eval_yaml_data[idx] = f"      TIME_STEP: {args.time_step}"
            if args.control_type == "dynamic":
                eval_yaml_data[idx] = "      TIME_STEP: 0.33"
        elif i.startswith("  SUCCESS_DISTANCE:"):
            eval_yaml_data[idx] = f"  SUCCESS_DISTANCE: {robot_goal}"
        elif i.startswith("    SUCCESS_DISTANCE:"):
            eval_yaml_data[idx] = f"    SUCCESS_DISTANCE: {robot_goal}"
        elif i.startswith("SEED:"):
            eval_yaml_data[idx] = f"SEED: {args.seed}"
        elif i.startswith("  DATA_PATH:"):
            if args.dataset == "ferst":
                data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/{split}/{split}.json.gz"
            elif args.dataset == "coda":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test/coda_spot_test.json.gz"
            elif args.dataset == "coda_straight_5m":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_5m/coda_spot_test_straight_5m.json.gz"
            elif args.dataset == "coda_straight_6m":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_6m/coda_spot_test_straight_6m.json.gz"
            elif args.dataset == "coda_straight_7m":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_7m/coda_spot_test_straight_7m.json.gz"
            elif args.dataset == "coda_straight_8m":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_8m/coda_spot_test_straight_8m.json.gz"
            elif args.dataset == "coda_straight_9m":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_9m/coda_spot_test_straight_9m.json.gz"
            elif args.dataset == "coda_straight_10m":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_10m/coda_spot_test_straight_10m.json.gz"
            elif args.dataset == "coda_straight_100m":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_spot_test_straight_100m/content/coda_spot_test_straight_100m.json.gz"
            elif args.dataset == "coda_lobby":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_lobby/coda_lobby.json.gz"
            elif args.dataset == "coda_lobby_hard":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_lobby_hard/coda_lobby_hard.json.gz"
            elif args.dataset == "google":
                data_path = "/coc/testnvme/jtruong33/data/datasets/google/val/content/boulder4772-2_v2.json.gz"
            elif args.dataset == "google_v3":
                data_path = "/coc/testnvme/jtruong33/data/datasets/google/val/content/boulder4772-2_v3.json.gz"
            elif args.dataset == "google_val":
                data_path = (
                    "/coc/testnvme/jtruong33/data/datasets/google/val_all/val.json.gz"
                )
            elif args.dataset == "ny":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/{split}/{split}.json.gz"
            elif args.dataset == "ny_train":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/train/train.json.gz"
            elif args.dataset == "hm3d_gibson_0.5":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson/pointnav_spot_0.5/{split}/{split}.json.gz"
            elif args.dataset == "uf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_unfurnished/{split}/{split}.json.gz"
            elif args.dataset == "hm3d_uf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d/pointnav_spot_unfurnished/{split}/{split}.json.gz"
            elif args.dataset == "ferst_mf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/multi_floor/{split}/{split}.json.gz"
            elif args.dataset == "ferst_mf_1":
                data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/multi_floor/val_1/val.json.gz"
            elif args.dataset == "ferst_stairs":
                data_path = "/coc/testnvme/jtruong33/data/datasets/ferst/stairs/{split}/{split}.json.gz"
            elif args.dataset == "hm3d_mf_only":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d/pointnav_spot_0.3_multi_floor_only/{split}/{split}.json.gz"
            elif args.dataset == "hm3d_straight":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d/pointnav_spot_0.4_single_floor_long/val/content/HY1NcmCgn3n.json.gz"
            elif args.dataset == "ny_google":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_spot_ny_google/val/val.json.gz"
            elif args.dataset == "google_1157":
                data_path = "/coc/testnvme/jtruong33/data/datasets/google/val_1157/content/mtv1157-1_lab.json.gz"
            elif args.dataset == "ny_mini":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson_ny/val_mini/val_mini.json.gz"
            eval_yaml_data[idx] = f"  DATA_PATH: {data_path}"
        elif i.startswith("      noise_multiplier:"):
            eval_yaml_data[idx] = f"      noise_multiplier: {args.noise_percent}"
    with open(new_eval_task_yaml_path, "w") as f:
        f.write("\n".join(eval_yaml_data))
    print("Created " + new_eval_task_yaml_path)

    # Edit the stored experiment yaml file
    with open(exp_yaml_path) as f:
        eval_exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(eval_exp_yaml_data):
        if i.startswith("BASE_TASK_CONFIG_PATH:"):
            eval_exp_yaml_data[
                idx
            ] = f"BASE_TASK_CONFIG_PATH: '{new_eval_task_yaml_path}'"
        elif i.startswith("TENSORBOARD_DIR:"):
            tb_dir = f"tb_eval_{args.control_type}"
            if args.ckpt != -1:
                tb_dir += f"_ckpt_{args.ckpt}"
            if args.video:
                tb_dir += "_video"
            eval_exp_yaml_data[
                idx
            ] = f"TENSORBOARD_DIR:    '{os.path.join(eval_dst_dir, 'tb_evals', tb_dir)}'"
        elif i.startswith("NUM_PROCESSES:"):
            eval_exp_yaml_data[idx] = "NUM_PROCESSES: 13"
        elif i.startswith("    name:"):
            if args.policy_name == "cnn":
                eval_exp_yaml_data[idx] = "    name: PointNavBaselinePolicy"
            if args.outdoor_nav:
                eval_exp_yaml_data[idx] = "    name: OutdoorPolicy"
            if args.context_map or args.context_waypoint or args.context_map_trajectory:
                eval_exp_yaml_data[idx] = "    name: PointNavContextPolicy"
            if (
                args.context_resnet_map
                or args.context_resnet_waypoint
                or args.context_resnet_map_trajectory
            ):
                eval_exp_yaml_data[idx] = "    name: PointNavResNetContextPolicy"
            if args.rnn_type == "TRANSFORMER":
                eval_exp_yaml_data[idx] = "    name: PointNavContextSMTPolicy"
        elif i.startswith("  COLLISION_PENALTY:"):
            eval_exp_yaml_data[idx] = f"  COLLISION_PENALTY: {args.collision_penalty}"
        elif i.startswith("CHECKPOINT_FOLDER:"):
            eval_exp_yaml_data[
                idx
            ] = f"CHECKPOINT_FOLDER:  '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("EVAL_CKPT_PATH_DIR:"):
            if args.ckpt == -1:
                eval_exp_yaml_data[
                    idx
                ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}'"
            else:
                eval_exp_yaml_data[
                    idx
                ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}/ckpt.{args.ckpt}.pth'"
        elif i.startswith("TXT_DIR:"):
            txt_dir = f"txts_eval_{args.control_type}"
            if args.ckpt != -1:
                txt_dir += f"_ckpt_{args.ckpt}"
            eval_exp_yaml_data[
                idx
            ] = f"TXT_DIR:            '{os.path.join(eval_dst_dir, 'txts', txt_dir)}'"
        elif i.startswith("VIDEO_OPTION:"):
            if args.video:
                eval_exp_yaml_data[idx] = "VIDEO_OPTION: ['disk']"
            else:
                eval_exp_yaml_data[idx] = "VIDEO_OPTION: []"
        elif i.startswith("NUM_ENVIRONMENTS:"):
            eval_exp_yaml_data[idx] = f"NUM_ENVIRONMENTS: {args.num_environments}"
            if (
                "ferst" in args.dataset
                or "coda" in args.dataset
                or args.dataset == "google"
            ):
                eval_exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
        elif i.startswith("SENSORS:"):
            if args.video:
                eval_exp_yaml_data[
                    idx
                ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR']"
            if args.use_gray:
                eval_exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
                if args.video:
                    eval_exp_yaml_data[
                        idx
                    ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
            elif args.use_gray_depth:
                eval_exp_yaml_data[
                    idx
                ] = "SENSORS: ['SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
                if args.video:
                    eval_exp_yaml_data[
                        idx
                    ] = "SENSORS: ['RGB_SENSOR', 'SPOT_LEFT_DEPTH_SENSOR', 'SPOT_RIGHT_DEPTH_SENSOR', 'SPOT_LEFT_GRAY_SENSOR', 'SPOT_RIGHT_GRAY_SENSOR']"
        elif i.startswith("VIDEO_DIR:"):
            video_dir = (
                "video_dir"
                if args.ckpt == -1
                else f"video_dir_{args.control_type}_ckpt_{args.ckpt}"
            )
            eval_exp_yaml_data[
                idx
            ] = f"VIDEO_DIR:          '{os.path.join(eval_dst_dir, 'videos', video_dir)}'"
        elif i.startswith("    SPLIT:"):
            if args.dataset == "hm3d":
                eval_exp_yaml_data[idx] = "    SPLIT: val"
            elif args.dataset == "ferst_20m":
                eval_exp_yaml_data[idx] = "    SPLIT: val_20m"
        elif i.startswith("      use_log_std:"):
            if args.log_std:
                eval_exp_yaml_data[idx] = "      use_log_std: True"
        elif i.startswith("    tgt_encoding:"):
            eval_exp_yaml_data[idx] = f"    tgt_encoding: '{args.target_encoding}'"
        elif i.startswith("    use_waypoint_encoder:"):
            if args.use_waypoint_encoder:
                eval_exp_yaml_data[idx] = f"    use_waypoint_encoder: True"
        elif i.startswith("    num_cnns:"):
            if args.two_cnns:
                eval_exp_yaml_data[idx] = "    num_cnns: 2"
        elif i.startswith("      ENABLED_TRANSFORMS: [ ]"):
            if args.pepper_noise:
                eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['PEPPER_NOISE']"
            elif args.cutout_noise:
                eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['CUTOUT']"
            elif args.median_blur:
                eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['MEDIAN_BLUR']"
        elif i.startswith("        NOISE_PERCENT:"):
            eval_exp_yaml_data[idx] = f"        NOISE_PERCENT: {args.noise_percent}"
        elif i.startswith("        KERNEL_SIZE:"):
            eval_exp_yaml_data[idx] = f"        KERNEL_SIZE: {args.kernel_size}"
        elif i.startswith("    context_hidden_size:"):
            eval_exp_yaml_data[
                idx
            ] = f"    context_hidden_size: {args.context_hidden_size}"
        elif i.startswith("    tgt_hidden_size:"):
            eval_exp_yaml_data[idx] = f"    tgt_hidden_size: {args.tgt_hidden_size}"
        elif i.startswith("    use_prev_action:"):
            if args.use_prev_action:
                eval_exp_yaml_data[idx] = f"    use_prev_action: True"
        elif i.startswith("    cnn_type:"):
            eval_exp_yaml_data[idx] = f"    cnn_type: '{args.cnn_type}'"
        elif i.startswith("    decoder_output:"):
            if args.splitnet and args.surface_normal:
                eval_exp_yaml_data[
                    idx
                ] = "    decoder_output: ['depth', 'surface_normals']"
        elif i.startswith("    visual_encoder:"):
            if args.splitnet:
                if args.visual_encoder == "shallow":
                    visual_encoder = "ShallowVisualEncoder"
                elif args.visual_encoder == "resnet":
                    visual_encoder = "BaseResNetEncoder"
                eval_exp_yaml_data[idx] = f"    visual_encoder: {visual_encoder}"
        elif i.startswith("    rnn_type:"):
            eval_exp_yaml_data[idx] = f"    rnn_type: {args.rnn_type}"
        elif i.startswith("    num_recurrent_layers:"):
            eval_exp_yaml_data[
                idx
            ] = f"    num_recurrent_layers: {args.num_recurrent_layers}"

    if os.path.isdir(tb_dir):
        response = input(
            f"{tb_dir} directory already exists. Delete, continue, or abort? [d/c/A]: "
        )
        if response == "d":
            print(f"Deleting {tb_dir}")
            shutil.rmtree(tb_dir)
        elif response == "c":
            print("Continuing.")
        else:
            print("Aborting.")
            exit()

    with open(new_eval_exp_yaml_path, "w") as f:
        f.write("\n".join(eval_exp_yaml_data))
    print("Created " + new_eval_exp_yaml_path)

    eval_experiment_name = experiment_name + exp_name

    # Create slurm job
    with open(EVAL_SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace("$TEMPLATE", eval_experiment_name)
        slurm_data = slurm_data.replace("$CONDA_ENV", CONDA_ENV)
        slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", HABITAT_LAB)
        slurm_data = slurm_data.replace(
            "$LOG", os.path.join(eval_dst_dir, eval_experiment_name)
        )
        slurm_data = slurm_data.replace("$CONFIG_YAML", new_eval_exp_yaml_path)
        slurm_data = slurm_data.replace("$PARTITION", args.partition)
        if args.partition == "overcap":
            slurm_data = slurm_data.replace("# ACCOUNT", "#SBATCH --account overcap")
    slurm_path = os.path.join(eval_dst_dir, eval_experiment_name + ".sh")
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
        f"\nSee output with:\ntail -F {os.path.join(eval_dst_dir, eval_experiment_name + '.err')}"
    )
