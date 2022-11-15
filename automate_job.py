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

parser.add_argument("-rotm", "--rotate-map", default=False, action="store_true")
parser.add_argument("-rpl", "--randomize-pitch-min", type=float, default=-1.0)
parser.add_argument("-rpu", "--randomize-pitch-max", type=float, default=1.0)

# Evaluation
parser.add_argument("-e", "--eval", default=False, action="store_true")
parser.add_argument("-cpt", "--ckpt", type=int, default=-1)
parser.add_argument("-v", "--video", default=False, action="store_true")
parser.add_argument("-d", "--debug", default=False, action="store_true")

parser.add_argument("-ngpu", "--num-gpus", type=int, default=8)
parser.add_argument("-x", default=False, action="store_true")
parser.add_argument("--ext", default="")
args = parser.parse_args()

EXP_YAML = "config/pointnav/ddppo_pointnav_spot.yaml"
TASK_YAML = "config/tasks/pointnav_spot.yaml"

experiment_name = args.experiment_name

dst_dir = os.path.join(RESULTS, experiment_name)
eval_dst_dir = os.path.join(RESULTS, experiment_name, "eval")

exp_yaml_path = os.path.join(HABITAT_BASELINES, EXP_YAML)
task_yaml_path = os.path.join(HABITAT_LAB, TASK_YAML)

new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_exp_yaml_path = os.path.join(dst_dir, os.path.basename(exp_yaml_path))

exp_name = ""

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
    eval_dst_dir = os.path.join(eval_dst_dir, args.dataset)

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

robot_urdf = "/coc/testnvme/jtruong33/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid_rot_fix.urdf"
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
        slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", BASE_DIR)
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
        if i.startswith("    robot:"):
            eval_yaml_data[idx] = f"    robot: '{args.robot}'"
        elif i.startswith(
            "      positon: [ -0.03740343144695029, 0.5, -0.4164822634134684 ]"
        ):
            position = (
                np.array([-0.03740343144695029, 0.5, -0.4164822634134684])
                * args.robot_scale
            )
            eval_yaml_data[idx] = f"      positon: {list(position)}"
        elif i.startswith(
            "      positon: [ 0.03614789234067159, 0.5, -0.4164822634134684 ]"
        ):
            position = (
                np.array([0.03614789234067159, 0.5, -0.4164822634134684])
                * args.robot_scale
            )
            eval_yaml_data[idx] = f"      positon: {list(position)}"
        elif i.startswith("      max_depth:"):
            max_depth = 3.5 * args.robot_scale
            eval_yaml_data[idx] = f"      max_depth: {max_depth:.2f}"
        elif i.startswith("        robot_urdf:"):
            eval_yaml_data[idx] = f"        robot_urdf: {robot_urdf}"
        elif i.startswith("      map_resolution:"):
            eval_yaml_data[idx] = f"      map_resolution: {args.map_resolution}"
            if args.rotate_map:
                eval_yaml_data[idx] = f"    ROTATE_MAP: True"
        elif i.startswith("        lin_vel_range:"):
            lin_vel = 0.5 * args.robot_scale
            eval_yaml_data[idx] = f"        lin_vel_range: [ -{lin_vel}, {lin_vel} ]"
        elif i.startswith("        min_rand_pitch:"):
            eval_yaml_data[idx] = f"        min_rand_pitch: {args.randomize_pitch_min}"
        elif i.startswith("        max_rand_pitch:"):
            eval_yaml_data[idx] = f"        max_rand_pitch: {args.randomize_pitch_max}"
        elif i.startswith("  seed:"):
            eval_yaml_data[idx] = f"  seed: {args.seed}"
        elif i.startswith("    success_distance:"):
            eval_yaml_data[idx] = f"    success_distance: 0.425"
        elif i.startswith("      success_distance:"):
            eval_yaml_data[idx] = f"      success_distance: 0.425"
        elif i.startswith("    data_path:"):
            if args.dataset == "ny":
                data_path = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/{split}/{split}.json.gz"
            elif args.dataset == "blender":
                data_path = (
                    "/coc/testnvme/jtruong33/data/datasets/blender/val/val.json.gz"
                )
            eval_yaml_data[idx] = f"    data_path: {data_path}"
    with open(new_eval_task_yaml_path, "w") as f:
        f.write("\n".join(eval_yaml_data))
    print("Created " + new_eval_task_yaml_path)

    # Edit the stored experiment yaml file
    with open(exp_yaml_path) as f:
        eval_exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(eval_exp_yaml_data):
        if i.startswith("  base_task_config_path:"):
            eval_exp_yaml_data[
                idx
            ] = f"  base_task_config_path: '{new_eval_task_yaml_path}'"
        elif i.startswith("  tensorboard_dir:"):
            tb_dir = f"tb_eval"
            if args.ckpt != -1:
                tb_dir += f"_ckpt_{args.ckpt}"
            if args.video:
                tb_dir += "_video"
            eval_exp_yaml_data[
                idx
            ] = f"  tensorboard_dir:    '{os.path.join(eval_dst_dir, 'tb_evals', tb_dir)}'"
        elif i.startswith("  num_environments:"):
            eval_exp_yaml_data[idx] = f"  num_environments: {args.num_environments}"
            if "ferst" in args.dataset or "coda" in args.dataset:
                eval_exp_yaml_data[idx] = "NUM_ENVIRONMENTS: 1"
        elif i.startswith("  video_dir:"):
            eval_exp_yaml_data[
                idx
            ] = f"  video_dir:          '{os.path.join(dst_dir, 'video_dir')}'"
        elif i.startswith("  eval_ckpt_path_dir:"):
            if args.ckpt == -1:
                eval_exp_yaml_data[
                    idx
                ] = f"  eval_ckpt_path_dir: '{os.path.join(dst_dir, 'checkpoints')}'"
            else:
                eval_exp_yaml_data[
                    idx
                ] = f"  eval_ckpt_path_dir: '{os.path.join(dst_dir, 'checkpoints')}/ckpt.{args.ckpt}.pth'"
        elif i.startswith("  checkpoint_folder:"):
            eval_exp_yaml_data[
                idx
            ] = f"  checkpoint_folder:  '{os.path.join(dst_dir, 'checkpoints')}'"
        elif i.startswith("  txt_dir:"):
            txt_dir = f"txts_eval"
            if args.ckpt != -1:
                txt_dir += f"_ckpt_{args.ckpt}"
            eval_exp_yaml_data[
                idx
            ] = f"  txt_dir:            '{os.path.join(eval_dst_dir, 'txts', txt_dir)}'"
        elif i.startswith("  video_option:"):
            if args.video:
                eval_exp_yaml_data[idx] = "  video_option: ['disk']"
            else:
                eval_exp_yaml_data[idx] = "  video_option: []"
        elif i.startswith("  sensors:"):
            if args.video:
                eval_exp_yaml_data[
                    idx
                ] = "  sensors: ['rgb_sensor', 'spot_left_depth_sensor', 'spot_right_depth_sensor']"

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
        slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", BASE_DIR)
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
