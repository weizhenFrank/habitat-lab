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
SLURM_TEMPLATE = os.path.join(HABITAT_LAB, "slurm_template_bc.sh")
EVAL_SLURM_TEMPLATE = os.path.join(HABITAT_LAB, "eval_slurm_template.sh")

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")

# Training
parser.add_argument("-sd", "--seed", type=int, default=1)
parser.add_argument("-ds", "--dataset", default="ny")
parser.add_argument("-ne", "--num_environments", type=int, default=64)
parser.add_argument(
    "-tf", "--teacher-force", default=False, action="store_true"
)
parser.add_argument("-bl", "--batch-length", type=int, default=8)
parser.add_argument("-sf", "--save-freq", type=int, default=100)
parser.add_argument("-msew", "--mse-weight", type=float, default=1.0)
parser.add_argument("-isw", "--is-weight", type=float, default=1.0)

parser.add_argument(
    "-cw", "--context-waypoint", default=False, action="store_true"
)
parser.add_argument(
    "-dwpt", "--debug-waypoint", default=False, action="store_true"
)
parser.add_argument("-r", "--regress", default="actions")
parser.add_argument("-cm", "--context-map", default=False, action="store_true")
parser.add_argument(
    "-crw", "--context-resnet-waypoint", default=False, action="store_true"
)
parser.add_argument(
    "-wpte", "--use-waypoint-encoder", default=False, action="store_true"
)
parser.add_argument(
    "-pp", "--use-pretrained-planner", default=False, action="store_true"
)
parser.add_argument(
    "-crm", "--context-resnet-map", default=False, action="store_true"
)
parser.add_argument("-mr", "--map-resolution", type=int, default=100)
parser.add_argument("-mpp", "--meters-per-pixel", type=float, default=0.5)
parser.add_argument(
    "-nrotm", "--no-rotate-map", default=False, action="store_true"
)
parser.add_argument("-csn", "--context-sensor-noise", type=float, default=0.0)
parser.add_argument("-lr", "--learning_rate", type=float, default=2.5e-4)
parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
parser.add_argument("-chs", "--context-hidden-size", type=int, default=512)
parser.add_argument("-ths", "--tgt-hidden-size", type=int, default=512)
parser.add_argument("-ncnn", "--num-cnns", type=int, default=0)
parser.add_argument("-cnnt", "--cnn-type", default="cnn_2d")
parser.add_argument("-rnnt", "--rnn-type", default="GRU")
parser.add_argument("-nrl", "--num_recurrent_layers", type=int, default=1)
parser.add_argument("-tgte", "--target_encoding", default="linear_2")
parser.add_argument(
    "-pa", "--use-prev-action", default=False, action="store_true"
)
parser.add_argument(
    "-sc", "--second-channel", default=False, action="store_true"
)
parser.add_argument(
    "-mc", "--multi-channel", default=False, action="store_true"
)
parser.add_argument(
    "-npg", "--noisy-pointgoal", default=False, action="store_true"
)
parser.add_argument("-cd", "--context-debug", default="")
parser.add_argument("--freeze", default=False, action="store_true")
parser.add_argument("-ft", "--finetune", default=False, action="store_true")
parser.add_argument(
    "-wpts", "--use-waypoint-student", default=False, action="store_true"
)
parser.add_argument(
    "-bs", "--use-baseline-student", default=False, action="store_true"
)
parser.add_argument("-cmse", "--clip-mse", default=False, action="store_true")
parser.add_argument("-l", "--loss", default="mse")
# Evaluation
parser.add_argument("-e", "--eval", default=False, action="store_true")
parser.add_argument("-cpt", "--ckpt", default="")
parser.add_argument("-v", "--video", default=False, action="store_true")
parser.add_argument("--ext", default="")

# Slurm
parser.add_argument("-p", "--partition", default="long")
parser.add_argument("-ngpu", "--num_gpus", type=int, default=1)
parser.add_argument("--constraint", default="x")
parser.add_argument("-x", default=False, action="store_true")

args = parser.parse_args()

EXP_YAML = "habitat_baselines/config/pointnav/behavioral_cloning.yaml"

TASK_YAML = "configs/tasks/pointnav_context_spot.yaml"

experiment_name = args.experiment_name

dst_dir = os.path.join(RESULTS, experiment_name)
eval_dst_dir = os.path.join(RESULTS, experiment_name, "eval/kinematic")

exp_yaml_path = os.path.join(HABITAT_LAB, EXP_YAML)
task_yaml_path = os.path.join(HABITAT_LAB, TASK_YAML)

new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_exp_yaml_path = os.path.join(dst_dir, os.path.basename(exp_yaml_path))

exp_name = f"_kinematic"

if args.eval:
    exp_name += f"_eval_{args.dataset}"
    eval_dst_dir += f"_{args.dataset}"
if args.ckpt != "":
    exp_name += f"_ckpt_{args.ckpt}"
    eval_dst_dir += f"_ckpt_{args.ckpt}"
if args.video:
    exp_name += "_video"
    eval_dst_dir += "_video"
if args.ext != "":
    exp_name += "_" + args.ext
    eval_dst_dir += "_" + args.ext

new_eval_task_yaml_path = (
    os.path.join(eval_dst_dir, os.path.basename(task_yaml_path)).split(
        ".yaml"
    )[0]
    + exp_name
    + ".yaml"
)
new_eval_exp_yaml_path = (
    os.path.join(eval_dst_dir, os.path.basename(exp_yaml_path)).split(".yaml")[
        0
    ]
    + exp_name
    + ".yaml"
)

# Training
if not args.eval:
    # Create directory
    if os.path.isdir(dst_dir):
        response = input(
            f"'{dst_dir}' already exists. Delete or abort? [d/A]: "
        )
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
    print(
        "Saved automate command: "
        + os.path.join(dst_dir, "automate_job_cmd.txt")
    )

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        task_yaml_data = f.read().splitlines()

    for idx, i in enumerate(task_yaml_data):
        if i.startswith("  SENSORS:"):
            pg = (
                "POINTGOAL_WITH_NOISY_GPS_COMPASS_SENSOR"
                if args.noisy_pointgoal
                else "POINTGOAL_WITH_GPS_COMPASS_SENSOR"
            )
            task_yaml_data[
                idx
            ] = f"  SENSORS: ['{pg}', 'CONTEXT_WAYPOINT_SENSOR', 'CONTEXT_MAP_SENSOR']"
        elif i.startswith("    BIN_POINTGOAL:"):
            if args.target_encoding == "ans_bin":
                task_yaml_data[idx] = f"    BIN_POINTGOAL: True"
        elif i.startswith("    MAP_RESOLUTION:"):
            task_yaml_data[idx] = f"    MAP_RESOLUTION: {args.map_resolution}"
        elif i.startswith("    METERS_PER_PIXEL:"):
            task_yaml_data[
                idx
            ] = f"    METERS_PER_PIXEL: {args.meters_per_pixel}"
        elif i.startswith("    ROTATE_MAP:"):
            if args.no_rotate_map:
                task_yaml_data[idx] = f"    ROTATE_MAP: False"
        elif i.startswith("    SECOND_CHANNEL:"):
            if args.second_channel:
                task_yaml_data[idx] = f"    SECOND_CHANNEL: True"
        elif i.startswith("    MULTI_CHANNEL:"):
            if args.multi_channel:
                task_yaml_data[idx] = f"    MULTI_CHANNEL: True"
        elif i.startswith("    DEBUG:"):
            task_yaml_data[idx] = f'    DEBUG: "{args.context_debug}"'
        elif i.startswith("SEED:"):
            task_yaml_data[idx] = f"SEED: {args.seed}"
        elif i.startswith("  DATA_PATH:"):
            if args.dataset == "ny":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/{split}/{split}.json.gz"
            elif args.dataset == "ny_1":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/train/content/wQN24R38a9N.json.gz"
            elif args.dataset == "coda_lobby":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_lobby/coda_lobby.json.gz"
            elif args.dataset == "google_1157":
                data_path = "/coc/testnvme/jtruong33/data/datasets/google/val_1157/content/mtv1157-1_lab.json.gz"
            elif args.dataset == "hm3d_mf":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d/pointnav_spot_0.3_multi_floor/{split}/{split}.json.gz"
            elif args.dataset == "ny_val":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/val/val.json.gz"
            task_yaml_data[idx] = f"  DATA_PATH: {data_path}"
    with open(new_task_yaml_path, "w") as f:
        f.write("\n".join(task_yaml_data))
    print("Created " + new_task_yaml_path)

    # Create experiment yaml file, using file within Habitat Lab repo as a template
    with open(exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(exp_yaml_data):
        if i.startswith("BASE_TASK_CONFIG_PATH:"):
            exp_yaml_data[
                idx
            ] = f"BASE_TASK_CONFIG_PATH: '{new_task_yaml_path}'"
        elif i.startswith("TOTAL_NUM_STEPS:"):
            exp_yaml_data[idx] = f"TOTAL_NUM_STEPS: 5e8"
        elif i.startswith("TENSORBOARD_DIR:"):
            exp_yaml_data[
                idx
            ] = f"TENSORBOARD_DIR:    '{os.path.join(dst_dir, 'tb')}'"
        elif i.startswith("NUM_ENVIRONMENTS:"):
            exp_yaml_data[idx] = f"NUM_ENVIRONMENTS: {args.num_environments}"
            if "ferst" in args.dataset or "coda" in args.dataset:
                exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
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
        elif i.startswith("  TEACHER_FORCE:"):
            if args.teacher_force:
                exp_yaml_data[idx] = "  TEACHER_FORCE: True"
        elif i.startswith("SL_LR:"):
            exp_yaml_data[idx] = f"SL_LR: {args.learning_rate}"
        elif i.startswith("SL_WD:"):
            exp_yaml_data[idx] = f"SL_WD: {args.weight_decay}"
        elif i.startswith("BATCH_LENGTH:"):
            exp_yaml_data[idx] = f"BATCH_LENGTH: {args.batch_length}"
        elif i.startswith("BATCHES_PER_CHECKPOINT:"):
            exp_yaml_data[idx] = f"BATCHES_PER_CHECKPOINT: {args.save_freq}"
        elif i.startswith("USE_WAYPOINT_STUDENT:"):
            if args.use_waypoint_student:
                exp_yaml_data[idx] = f"USE_WAYPOINT_STUDENT: True"
        elif i.startswith("USE_BASELINE_STUDENT:"):
            if args.use_baseline_student:
                exp_yaml_data[idx] = f"USE_BASELINE_STUDENT: True"
        elif i.startswith("DEBUG_WAYPOINT:"):
            if args.debug_waypoint:
                exp_yaml_data[idx] = f"DEBUG_WAYPOINT: True"
        elif i.startswith("CLIP_MSE:"):
            if args.clip_mse:
                exp_yaml_data[idx] = f"CLIP_MSE: True"
        elif i.startswith("LOSS:"):
            exp_yaml_data[idx] = f"LOSS: {args.loss}"
        elif i.startswith("REGRESS:"):
            exp_yaml_data[idx] = f"REGRESS: {args.regress}"
        elif i.startswith("FREEZE_POLICY:"):
            if args.freeze:
                exp_yaml_data[idx] = f"FREEZE_POLICY: True"
        elif i.startswith("MSE_WEIGHT:"):
            exp_yaml_data[idx] = f"MSE_WEIGHT: {args.mse_weight}"
        elif i.startswith("IS_WEIGHT:"):
            exp_yaml_data[idx] = f"IS_WEIGHT: {args.is_weight}"
        elif i.startswith("    num_cnns:"):
            exp_yaml_data[idx] = f"    num_cnns: {args.num_cnns}"
        elif i.startswith("    tgt_encoding:"):
            exp_yaml_data[idx] = f"    tgt_encoding: '{args.target_encoding}'"
        elif i.startswith("    use_waypoint_encoder:"):
            if args.use_waypoint_encoder:
                exp_yaml_data[idx] = f"    use_waypoint_encoder: True"
        elif i.startswith("    use_pretrained_planner:"):
            if args.use_pretrained_planner:
                exp_yaml_data[idx] = f"    use_pretrained_planner: True"
        elif i.startswith("    context_hidden_size:"):
            exp_yaml_data[
                idx
            ] = f"    context_hidden_size: {args.context_hidden_size}"
        elif i.startswith("    tgt_hidden_size:"):
            exp_yaml_data[idx] = f"    tgt_hidden_size: {args.tgt_hidden_size}"
        elif i.startswith("    use_prev_action:"):
            if args.use_prev_action:
                exp_yaml_data[idx] = f"    use_prev_action: True"
        elif i.startswith("    cnn_type:"):
            exp_yaml_data[idx] = f"    cnn_type: '{args.cnn_type}'"
        elif i.startswith("    rnn_type:"):
            exp_yaml_data[idx] = f"    rnn_type: '{args.rnn_type}'"
        elif i.startswith("    num_recurrent_layers:"):
            exp_yaml_data[
                idx
            ] = f"    num_recurrent_layers: '{args.num_recurrent_layers}'"
        elif i.startswith("    name:"):
            if args.context_waypoint or args.context_map:
                exp_yaml_data[idx] = "    name: PointNavContextPolicy"
            if args.context_resnet_waypoint or args.context_resnet_map:
                exp_yaml_data[idx] = "    name: PointNavResNetContextPolicy"
            if args.use_baseline_student:
                exp_yaml_data[idx] = "    name: PointNavBaselinePolicy"
        elif i.startswith("      ENABLED_TRANSFORMS: [ ]"):
            if args.pepper_noise:
                exp_yaml_data[
                    idx
                ] = "      ENABLED_TRANSFORMS: ['PEPPER_NOISE']"
            elif args.cutout_noise:
                exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: ['CUTOUT']"
            elif args.median_blur:
                exp_yaml_data[
                    idx
                ] = "      ENABLED_TRANSFORMS: ['MEDIAN_BLUR']"
        elif i.startswith("    pretrained_weights: "):
            if args.finetune:
                #         # ft_weights = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_simple_cnn_cutout_nhy_2hz_ny_rand_pitch_bp_0.03_sd_1/checkpoints/ckpt.84.pth"
                #         # ft_weights = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_simple_cnn_cutout_nhy_2hz_hm3d_mf_rand_pitch_-1.0_1.0_bp_0.03_log_sd_1/checkpoints/ckpt.95.pth"
                ft_weights = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_simple_cnn_cutout_nhy_2hz_ny_rand_pitch_-1.0_1.0_bp_0.03_sd_1_16env_context_no_noise_log_agent_rot/checkpoints/ckpt.94.pth"
                exp_yaml_data[idx] = f"    pretrained_weights: {ft_weights}"
        elif i.startswith("    pretrained: "):
            if args.finetune:
                exp_yaml_data[idx] = f"    pretrained: True"
    with open(new_exp_yaml_path, "w") as f:
        f.write("\n".join(exp_yaml_data))
    print("Created " + new_exp_yaml_path)

    # Create slurm job
    with open(SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace("$TEMPLATE", experiment_name)
        slurm_data = slurm_data.replace("$CONDA_ENV", CONDA_ENV)
        slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", HABITAT_LAB)
        slurm_data = slurm_data.replace(
            "$LOG", os.path.join(dst_dir, experiment_name)
        )
        slurm_data = slurm_data.replace("$CONFIG_YAML", new_exp_yaml_path)
        slurm_data = slurm_data.replace("$PARTITION", args.partition)
        slurm_data = slurm_data.replace("$GPUS", f"{args.num_gpus}")
    if args.constraint == "6000_a40":
        slurm_data = slurm_data.replace(
            "# CONSTRAINT", "#SBATCH --constraint rtx_6000|a40"
        )
    elif args.constraint == "6000":
        slurm_data = slurm_data.replace(
            "# CONSTRAINT", "#SBATCH --constraint rtx_6000"
        )
    elif args.constraint == "a40":
        slurm_data = slurm_data.replace(
            "# CONSTRAINT", "#SBATCH --constraint a40"
        )
    slurm_path = os.path.join(dst_dir, experiment_name + ".sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_data)
    print("Generated slurm job: " + slurm_path)

    if not args.x:
        # Submit slurm job
        cmd = "sbatch " + slurm_path
        subprocess.check_call(cmd.split(), cwd=dst_dir)
        print(
            f"\nSee output with:\ntail -F {os.path.join(dst_dir, experiment_name + '.err')}"
        )
    else:
        print(slurm_data)

# Evaluation
else:
    # Make sure folder exists
    assert os.path.isdir(dst_dir), f"{dst_dir} directory does not exist"
    os.makedirs(eval_dst_dir, exist_ok=True)
    with open(os.path.join(eval_dst_dir, "automate_job_cmd.txt"), "w") as f:
        f.write(automate_command)
    print(
        "Saved automate command: "
        + os.path.join(eval_dst_dir, "automate_job_cmd.txt")
    )

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        eval_yaml_data = f.read().splitlines()

    for idx, i in enumerate(eval_yaml_data):
        if i.startswith("  SENSORS:"):
            pg = (
                "POINTGOAL_WITH_NOISY_GPS_COMPASS_SENSOR"
                if args.noisy_pointgoal
                else "POINTGOAL_WITH_GPS_COMPASS_SENSOR"
            )
            eval_yaml_data[idx] = f"  SENSORS: ['{pg}']"
            if args.context_waypoint and args.context_map:
                eval_yaml_data[
                    idx
                ] = f"  SENSORS: ['{pg}', 'CONTEXT_WAYPOINT_SENSOR', 'CONTEXT_MAP_SENSOR']"
            elif args.context_waypoint or args.context_resnet_waypoint:
                eval_yaml_data[
                    idx
                ] = f"  SENSORS: ['{pg}', 'CONTEXT_WAYPOINT_SENSOR']"
            elif args.context_map or args.context_resnet_map:
                eval_yaml_data[
                    idx
                ] = f"  SENSORS: ['{pg}', 'CONTEXT_MAP_SENSOR']"
        elif i.startswith("  SUCCESS_DISTANCE:"):
            eval_yaml_data[idx] = f"  SUCCESS_DISTANCE: 0.425"
        elif i.startswith("    SUCCESS_DISTANCE:"):
            eval_yaml_data[idx] = f"    SUCCESS_DISTANCE: 0.425"
        elif i.startswith("    BIN_POINTGOAL:"):
            if args.target_encoding == "ans_bin":
                eval_yaml_data[idx] = f"    BIN_POINTGOAL: True"
        elif i.startswith("    MAP_RESOLUTION:"):
            eval_yaml_data[idx] = f"    MAP_RESOLUTION: {args.map_resolution}"
        elif i.startswith("    METERS_PER_PIXEL:"):
            eval_yaml_data[
                idx
            ] = f"    METERS_PER_PIXEL: {args.meters_per_pixel}"
        elif i.startswith("    ROTATE_MAP:"):
            if args.no_rotate_map:
                eval_yaml_data[idx] = f"    ROTATE_MAP: True"
        elif i.startswith("    SECOND_CHANNEL:"):
            if args.second_channel:
                eval_yaml_data[idx] = f"    SECOND_CHANNEL: True"
        elif i.startswith("    MULTI_CHANNEL:"):
            if args.multi_channel:
                eval_yaml_data[idx] = f"    MULTI_CHANNEL: True"
        elif i.startswith("    DEBUG:"):
            eval_yaml_data[idx] = f'    DEBUG: "{args.context_debug}"'
        elif i.startswith("      MIN_RAND_PITCH:"):
            eval_yaml_data[idx] = f"      MIN_RAND_PITCH: 0.0"
        elif i.startswith("      MAX_RAND_PITCH:"):
            eval_yaml_data[idx] = f"      MAX_RAND_PITCH: 0.0"
        elif i.startswith("SEED:"):
            eval_yaml_data[idx] = f"SEED: {args.seed}"
        elif i.startswith("  DATA_PATH:"):
            if args.dataset == "google":
                data_path = "/coc/testnvme/jtruong33/data/datasets/google/val/content/boulder4772-2_v2.json.gz"
            elif args.dataset == "google_v3":
                data_path = "/coc/testnvme/jtruong33/data/datasets/google/val/content/boulder4772-2_v3.json.gz"
            elif args.dataset == "google_val":
                data_path = "/coc/testnvme/jtruong33/data/datasets/google/val_all/val.json.gz"
            elif args.dataset == "ny_val":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/val/val.json.gz"
            elif args.dataset == "ny_mini":
                data_path = "/coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson_ny/val_mini/val_mini.json.gz"
            elif args.dataset == "ny_train":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/train/train.json.gz"
            elif args.dataset == "ny_1":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/train/content/wQN24R38a9N.json.gz"
            elif args.dataset == "coda_lobby":
                data_path = "/coc/testnvme/jtruong33/data/datasets/coda/coda_lobby/coda_lobby.json.gz"
            elif args.dataset == "google_1157":
                data_path = "/coc/testnvme/jtruong33/data/datasets/google/val_1157/content/mtv1157-1_lab.json.gz"
            eval_yaml_data[idx] = f"  DATA_PATH: {data_path}"
        elif i.startswith("      noise_multiplier:"):
            eval_yaml_data[
                idx
            ] = f"      noise_multiplier: {args.noise_percent}"
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
            tb_dir = "tb_eval_kinematic"
            if args.ckpt != "":
                tb_dir += f"_ckpt_{args.ckpt}"
            if args.video:
                tb_dir += "_video"
            eval_exp_yaml_data[
                idx
            ] = f"TENSORBOARD_DIR:    '{os.path.join(eval_dst_dir, 'tb_evals', tb_dir)}'"
        elif i.startswith("    name:"):
            if args.context_waypoint or args.context_map:
                eval_exp_yaml_data[idx] = "    name: PointNavContextBCPolicy"
            if args.context_resnet_waypoint or args.context_resnet_map:
                eval_exp_yaml_data[
                    idx
                ] = "    name: PointNavResNetContextPolicy"
            if args.use_baseline_student:
                eval_exp_yaml_data[idx] = "    name: PointNavBaselinePolicy"
        elif i.startswith("CHECKPOINT_FOLDER:"):
            eval_exp_yaml_data[
                idx
            ] = f"CHECKPOINT_FOLDER:  '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("EVAL_CKPT_PATH_DIR:"):
            if args.ckpt == "":
                eval_exp_yaml_data[
                    idx
                ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}'"
            else:
                eval_exp_yaml_data[
                    idx
                ] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir, 'checkpoints')}/{args.ckpt}.pth'"
        elif i.startswith("TXT_DIR:"):
            txt_dir = f"txts_eval_kinematic"
            if args.ckpt != "":
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
            eval_exp_yaml_data[
                idx
            ] = f"NUM_ENVIRONMENTS: {args.num_environments}"
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
        elif i.startswith("VIDEO_DIR:"):
            video_dir = (
                "video_dir"
                if args.ckpt == ""
                else f"video_dir_kinematic_ckpt_{args.ckpt}"
            )
            eval_exp_yaml_data[
                idx
            ] = f"VIDEO_DIR:          '{os.path.join(eval_dst_dir, 'videos', video_dir)}'"
        elif i.startswith("    num_cnns:"):
            eval_exp_yaml_data[idx] = f"    num_cnns: {args.num_cnns}"
        elif i.startswith("    tgt_encoding:"):
            eval_exp_yaml_data[
                idx
            ] = f"    tgt_encoding: '{args.target_encoding}'"
        elif i.startswith("      ENABLED_TRANSFORMS:"):
            eval_exp_yaml_data[idx] = "      ENABLED_TRANSFORMS: []"
        elif i.startswith("    context_hidden_size:"):
            eval_exp_yaml_data[
                idx
            ] = f"    context_hidden_size: {args.context_hidden_size}"
        elif i.startswith("    tgt_hidden_size:"):
            eval_exp_yaml_data[
                idx
            ] = f"    tgt_hidden_size: {args.tgt_hidden_size}"
        elif i.startswith("    use_prev_action:"):
            if args.use_prev_action:
                eval_exp_yaml_data[idx] = f"    use_prev_action: True"
        elif i.startswith("    cnn_type:"):
            eval_exp_yaml_data[idx] = f"    cnn_type: '{args.cnn_type}'"
        elif i.startswith("    use_waypoint_encoder:"):
            if args.use_waypoint_encoder:
                eval_exp_yaml_data[idx] = f"    use_waypoint_encoder: True"
        elif i.startswith("    use_pretrained_planner:"):
            if args.use_pretrained_planner:
                eval_exp_yaml_data[idx] = f"    use_pretrained_planner: True"

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
            slurm_data = slurm_data.replace(
                "# ACCOUNT", "#SBATCH --account overcap"
            )
    slurm_path = os.path.join(eval_dst_dir, eval_experiment_name + ".sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_data)
    print("Generated slurm job: " + slurm_path)

    if not args.x:
        # Submit slurm job
        cmd = "sbatch " + slurm_path
        subprocess.check_call(cmd.split(), cwd=dst_dir)
        print(
            f"\nSee output with:\ntail -F {os.path.join(eval_dst_dir, eval_experiment_name + '.err')}"
        )
    else:
        print(slurm_data)
