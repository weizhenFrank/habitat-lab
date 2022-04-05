"""
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
"""
import argparse
import os
import subprocess
import sys

automate_command = "python " + " ".join(sys.argv)
IGIBSON = "/coc/testnvme/jtruong33/igibson_dyn/iGibson/igibson"
HABITAT = "/coc/testnvme/jtruong33/igibson_dyn/habitat-lab"
RESULTS = "/coc/pskynet3/jtruong33/develop/flash_results/igibson_dyn_results"
URDFS = "/coc/testnvme/jtruong33/data/URDF_demo_assets"
EVAL_SLURM_TEMPLATE = os.path.join(HABITAT, "eval_slurm_template.sh")


parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")

# Evaluation
parser.add_argument("-sd", "--seed", type=int, default=100)
parser.add_argument("-r", "--robot", default="aliengo")
parser.add_argument("-c", "--control-type", default="kinematic")
parser.add_argument("-p", "--partition", default="long")
parser.add_argument("-d", "--debug", default=False, action="store_true")
parser.add_argument("-ts", "--time-step", type=float, default=1.0)

parser.add_argument("-cpt", "--ckpt", type=int, default=-1)
parser.add_argument("-v", "--video", default=False, action="store_true")
parser.add_argument("-n", "--noise", default=False, action="store_true")
parser.add_argument("-x", default=False, action="store_true")
parser.add_argument("--ext", default="")

args = parser.parse_args()


EXP_YAML = "habitat_baselines/config/pointnav/ddppo_pointnav_quadruped_eval.yaml"
TASK_YAML = "configs/tasks/pointnav_quadruped_eval.yaml"

experiment_name = args.experiment_name

dst_dir = os.path.join(RESULTS, experiment_name)

dir_name = "{}_habitat".format(args.control_type)

eval_name = "_{}_habitat_eval".format(args.control_type)
if args.ckpt != -1:
    eval_name += "_ckpt_{}".format(args.ckpt)
    dir_name += "_ckpt_{}".format(args.ckpt)
if args.video:
    eval_name += "_video"
    dir_name += "_video"
if args.noise:
    eval_name += "_noise"
    dir_name += "_noise"
if args.ext != "":
    eval_name += "_" + args.ext
    dir_name += "_" + args.ext

eval_dst_dir = os.path.join(RESULTS, experiment_name, "eval", dir_name)

task_yaml_path = os.path.join(HABITAT, TASK_YAML)
exp_yaml_path = os.path.join(HABITAT, EXP_YAML)

new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_exp_yaml_path = os.path.join(dst_dir, os.path.basename(exp_yaml_path))


new_eval_task_yaml_path = (
    os.path.join(eval_dst_dir, os.path.basename(task_yaml_path)).split(".yaml")[0]
    + eval_name
    + ".yaml"
)
new_eval_exp_yaml_path = (
    os.path.join(eval_dst_dir, os.path.basename(exp_yaml_path)).split(".yaml")[0]
    + eval_name
    + ".yaml"
)

robot_goal_dict = {
    "a1": 0.24,
    "aliengo": 0.3235,
    "spot": 0.425,
}

robot_vel_dict = {
    "a1": [[-0.23, 0.23], [-8.02, 8.02]],
    "aliengo": [[-0.28, 0.28], [-9.74, 9.74]],
    "spot": [[-0.5, 0.5], [-17.19, 17.19]],
}

robot_urdfs_dict = {
    "a1": os.path.join(URDFS, "a1/a1.urdf"),
    "aliengo": os.path.join(URDFS, "aliengo/urdf/aliengo.urdf"),
    "daisy": os.path.join(URDFS, "daisy/daisy_advanced_akshara.urdf"),
    "spot": os.path.join(
        URDFS, "spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf"
    ),
    "locobot": os.path.join(URDFS, "locobot/urdf/locobot_description2.urdf"),
}

num_steps_dict = {
    "a1": 326,
    "aliengo": 268,
    "spot": 150,
}

robot_radius_dict = {"a1": 0.2, "aliengo": 0.22, "spot": 0.3}

succ_radius = robot_goal_dict[args.robot.lower()]
robot_lin_vel, robot_ang_vel = robot_vel_dict[args.robot.lower()]
robot_urdf = robot_urdfs_dict[args.robot.lower()]
robot_radius = robot_radius_dict[args.robot.lower()]
robot_num_steps = num_steps_dict[args.robot.lower()]

assert os.path.isdir(dst_dir), "{} directory does not exist".format(dst_dir)
os.makedirs(eval_dst_dir, exist_ok=True)
with open(os.path.join(eval_dst_dir, "automate_job_cmd.txt"), "w") as f:
    f.write(automate_command)
print("Saved automate command: " + os.path.join(eval_dst_dir, "automate_job_cmd.txt"))

# Create task yaml file, using file within Habitat Lab repo as a template
with open(task_yaml_path) as f:
    eval_task_yaml_data = f.read().splitlines()

for idx, i in enumerate(eval_task_yaml_data):
    if i.startswith("SEED:"):
        eval_task_yaml_data[idx] = "SEED: {}".format(args.seed)
    elif i.startswith("  MAX_EPISODE_STEPS:"):
        eval_task_yaml_data[idx] = "  MAX_EPISODE_STEPS: {}".format(robot_num_steps)
    elif i.startswith("    RADIUS:"):
        eval_task_yaml_data[idx] = "    RADIUS: {}".format(robot_radius)
    elif i.startswith("  SUCCESS_DISTANCE:"):
        eval_task_yaml_data[idx] = "  SUCCESS_DISTANCE: {}".format(succ_radius)
    elif i.startswith("  POSSIBLE_ACTIONS:"):
        if args.control_type == "dynamic":
            control_type = "DYNAMIC_VELOCITY_CONTROL"
        else:
            control_type = "VELOCITY_CONTROL"
        eval_task_yaml_data[idx] = '  POSSIBLE_ACTIONS: ["{}"]'.format(control_type)
    elif i.startswith("    VELOCITY_CONTROL:"):
        if args.control_type == "dynamic":
            eval_task_yaml_data[idx] = "    DYNAMIC_VELOCITY_CONTROL:"
    elif i.startswith("    DYNAMIC_VELOCITY_CONTROL:"):
        if not args.control_type == "dynamic":
            eval_task_yaml_data[idx] = "    VELOCITY_CONTROL:"
    elif i.startswith("      TIME_STEP:"):
        if args.control_type == "dynamic":
            eval_task_yaml_data[idx] = "      TIME_STEP: 0.33"
    elif i.startswith("      LIN_VEL_RANGE:"):
        eval_task_yaml_data[idx] = "      LIN_VEL_RANGE: [{}, {}]".format(
            robot_lin_vel[0], robot_lin_vel[1]
        )
    elif i.startswith("      ANG_VEL_RANGE:"):
        eval_task_yaml_data[idx] = "      ANG_VEL_RANGE: [{}, {}]".format(
            robot_ang_vel[0], robot_ang_vel[1]
        )
    elif i.startswith("      HOR_VEL_RANGE:"):
        eval_task_yaml_data[idx] = "      HOR_VEL_RANGE: [{}, {}]".format(
            robot_lin_vel[0], robot_lin_vel[1]
        )
    elif i.startswith("  ROBOT:"):
        eval_task_yaml_data[idx] = "  ROBOT: {}".format(args.robot)
    elif i.startswith("      NOISE_MEAN_X:"):
        if args.noise:
            eval_task_yaml_data[idx] = "      NOISE_MEAN_X: {}".format(-0.0171)
    elif i.startswith("      NOISE_MEAN_Y:"):
        if args.noise:
            eval_task_yaml_data[idx] = "      NOISE_MEAN_Y: {}".format(0.0141)
    elif i.startswith("      NOISE_VAR_X:"):
        if args.noise:
            eval_task_yaml_data[idx] = "      NOISE_VAR_X: {}".format(0.0439)
    elif i.startswith("      NOISE_VAR_Y:"):
        if args.noise:
            eval_task_yaml_data[idx] = "      NOISE_VAR_Y: {}".format(0.0282)
    elif i.startswith("      ROBOT_URDF:"):
        eval_task_yaml_data[idx] = "      ROBOT_URDF: {}".format(robot_urdf)
    elif i.startswith("    SUCCESS_DISTANCE:"):
        eval_task_yaml_data[idx] = "    SUCCESS_DISTANCE: {}".format(succ_radius)
    # elif i.startswith("  DATA_PATH:"):
    #     eval_task_yaml_data[
    #         idx
    #     ] = "  DATA_PATH: /coc/testnvme/jtruong33/data/datasets/pointnav_hm3d_gibson/pointnav_{}_{}/{{split}}/{{split}}.json.gz".format(
    #         args.robot.lower(), robot_radius
    #     )

with open(new_eval_task_yaml_path, "w") as f:
    f.write("\n".join(eval_task_yaml_data))
print("Created " + new_eval_task_yaml_path)

# Create experiment yaml file, using file within Habitat Lab repo as a template
with open(exp_yaml_path) as f:
    eval_exp_yaml_data = f.read().splitlines()

for idx, i in enumerate(eval_exp_yaml_data):
    if i.startswith("BASE_TASK_CONFIG_PATH:"):
        eval_exp_yaml_data[idx] = "BASE_TASK_CONFIG_PATH: '{}'".format(
            new_eval_task_yaml_path
        )
    elif i.startswith("TENSORBOARD_DIR:"):
        tb_dir = "tb_eval_{}".format(args.control_type)
        if args.ckpt != -1:
            tb_dir += "_ckpt_{}".format(args.ckpt)
        if args.video:
            tb_dir += "_video"
        eval_exp_yaml_data[idx] = "TENSORBOARD_DIR:    '{}'".format(
            os.path.join(eval_dst_dir, "tb_evals", tb_dir)
        )
    elif i.startswith("VIDEO_DIR:"):
        video_dir = (
            "video_dir"
            if args.ckpt == -1
            else "video_dir_{}_ckpt_{}".format(args.control_type, args.ckpt)
        )
        eval_exp_yaml_data[idx] = "VIDEO_DIR:          '{}'".format(
            os.path.join(eval_dst_dir, "videos", video_dir)
        )
    elif i.startswith("EVAL_CKPT_PATH_DIR:"):
        if args.ckpt == -1:
            eval_exp_yaml_data[idx] = "EVAL_CKPT_PATH_DIR: '{}'".format(
                os.path.join(dst_dir, "checkpoints")
            )
        else:
            eval_exp_yaml_data[idx] = "EVAL_CKPT_PATH_DIR: '{}/ckpt.{}.pth'".format(
                os.path.join(dst_dir, "checkpoints"), args.ckpt
            )
    elif i.startswith("CHECKPOINT_FOLDER:"):
        eval_exp_yaml_data[idx] = "CHECKPOINT_FOLDER:  '{}'".format(
            os.path.join(dst_dir, "checkpoints")
        )
    elif i.startswith("TXT_DIR:"):
        txt_dir = "txts_eval_{}".format(args.control_type)
        if args.ckpt != -1:
            txt_dir += "_ckpt_{}".format(args.ckpt)
        eval_exp_yaml_data[idx] = "TXT_DIR:            '{}'".format(
            os.path.join(eval_dst_dir, "txts", txt_dir)
        )
    elif i.startswith("VIDEO_OPTION:"):
        if args.video:
            eval_exp_yaml_data[idx] = "VIDEO_OPTION: ['disk']"
        else:
            eval_exp_yaml_data[idx] = "VIDEO_OPTION: []"
    elif i.startswith("SENSORS:"):
        eval_exp_yaml_data[idx] = "SENSORS: ['DEPTH_SENSOR']"
        if args.video:
            eval_exp_yaml_data[idx] = "SENSORS: ['RGB_SENSOR','DEPTH_SENSOR']"
with open(new_eval_exp_yaml_path, "w") as f:
    f.write("\n".join(eval_exp_yaml_data))
print("Created " + new_eval_exp_yaml_path)

eval_experiment_name = experiment_name + eval_name
# Create slurm job
with open(EVAL_SLURM_TEMPLATE) as f:
    slurm_data = f.read()
    slurm_data = slurm_data.replace("$TEMPLATE", eval_experiment_name)
    slurm_data = slurm_data.replace(
        "$LOG", os.path.join(eval_dst_dir, eval_experiment_name)
    )
    slurm_data = slurm_data.replace("$HABITAT_REPO_PATH", HABITAT)
    slurm_data = slurm_data.replace("$CONFIG_YAML", new_eval_exp_yaml_path)
    slurm_data = slurm_data.replace("$PARTITION", args.partition)
slurm_path = os.path.join(eval_dst_dir, eval_experiment_name + ".sh")
with open(slurm_path, "w") as f:
    f.write(slurm_data)
print("Generated slurm job: " + slurm_path)

if not args.x:
    # Submit slurm job
    cmd = "sbatch " + slurm_path
    subprocess.check_call(cmd.split(), cwd=eval_dst_dir)
else:
    print(slurm_data)

print(
    "\nSee output with:\ntail -F {}".format(
        os.path.join(eval_dst_dir, eval_experiment_name + ".err")
    )
)
