import argparse
import os
import random

import numpy as np
import torch
from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.behavioral_cloning.agents import (
    MapStudent,
    WaypointTeacher,
)
from habitat_baselines.utils.common import (
    action_to_velocity_control,
    batch_obs,
)
from habitat_baselines.utils.env_utils import construct_envs
from skimage.draw import disk

CHECKPOINT_PATH = ""
LAST_TEACHER_BATCH = 1000


@baseline_registry.register_trainer(name="data_collector")
class DataCollector(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None, eval=False):
        logger.info(f"env config: {config}")
        self.eval = eval
        if self.eval:
            config.defrost()
            # config.TASK_CONFIG.DATASET.DATA_PATH = "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/val_1157/val.json.gz"
            config.TASK_CONFIG.DATASET.DATA_PATH = "/coc/testnvme/jtruong33/data/datasets/google/val_1157/content/mtv1157-1_lab.json.gz"
            config.freeze()

        self.config = config
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        # self.teacher = WaypointTeacher(self.config.RL)
        self.teacher = MapStudent(self.config)

        self.obs_transforms = get_active_obs_transforms(self.config)

    def train(self):
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        observations = self.envs.reset()

        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # Teacher tensors
        teacher_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.teacher.actor_critic.net.num_recurrent_layers,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )
        num_actions = 2
        teacher_prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            num_actions,
            device=self.device,
            dtype=torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            dtype=torch.bool,
            device=self.device,
        )

        self.teacher.actor_critic.eval()

        context_maps = []
        context_waypoints = []
        context_goals = []
        context_waypoint_maps = []
        # n_iter = 10000 if self.eval else 2000000
        # n_iter = 200000
        # n_iter = 100000
        # n_iter = 50000
        n_iter = 1000
        print("N_ITER: ", n_iter)
        for iteration in range(1, n_iter):
            print(f"# iter: {iteration}, {n_iter}")
            # for iteration in range(1, 100):
            with torch.no_grad():
                (
                    _,
                    teacher_actions,
                    _,
                    teacher_hidden_states,
                ) = self.teacher.actor_critic.act(
                    batch,
                    teacher_hidden_states,
                    teacher_prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

            ## teacher drives
            execute_actions = teacher_actions
            # collect data w/ rotated map
            # execute_actions[0, 1] = np.random.uniform(-1, 1)
            teacher_prev_actions.copy_(execute_actions)
            step_actions = [
                action_to_velocity_control(a, "VELOCITY_CONTROL")
                for a in execute_actions.to(device="cpu")
            ]
            outputs = self.envs.step(step_actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            for o in observations:
                # if np.exp(o["pointgoal_with_gps_compass"][0]) > 2:
                for k, v in o.items():
                    if k == "context_map":
                        context_maps.append(v)
                    elif k == "context_waypoint":
                        # returns r, theta
                        mid = 128
                        mpp = 0.1
                        waypoint_map = np.zeros((256, 256))
                        r, theta = v
                        x = (np.exp(r) / mpp) * np.cos(theta)
                        y = (np.exp(r) / mpp) * np.sin(theta)

                        row, col = np.clip(int(mid - x), 5, 250), np.clip(
                            int(mid - y), 5, 250
                        )
                        rr, cc = disk((row, col), 5)
                        waypoint_map[rr, cc] = 1.0
                        context_waypoints.append(np.array([r, theta]))
                        context_waypoint_maps.append(waypoint_map)
                    elif k == "pointgoal_with_gps_compass":
                        mid = 128
                        mpp = 0.1
                        goal_map = np.zeros((256, 256))
                        rr, cc = disk((128, 128), 5)
                        goal_map[rr, cc] = 1.0
                        r, theta = v
                        x = (np.exp(r) / mpp) * np.cos(theta)
                        y = (np.exp(r) / mpp) * np.sin(theta)

                        row, col = np.clip(int(mid - x), 5, 250), np.clip(
                            int(mid - y), 5, 250
                        )

                        rr, cc = disk((row, col), 5)
                        goal_map[rr, cc] = 1.0
                        context_goals.append(goal_map)

            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )
        print("# maps: ", np.array(context_maps).shape)
        base_pth = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/"
        prefix = "eval_" if self.eval else ""
        np.save(
            os.path.join(
                base_pth, prefix + "context_maps_1157_student_3m.npy"
            ),
            np.array(context_maps),
        )
        np.save(
            os.path.join(
                base_pth,
                prefix + "context_waypoint_maps_1157_student_3m.npy",
            ),
            np.array(context_waypoint_maps),
        )
        np.save(
            os.path.join(
                base_pth, prefix + "context_waypoints_1157_student_3m.npy"
            ),
            np.array(context_waypoints),
        )
        np.save(
            os.path.join(
                base_pth, prefix + "context_goals_1157_student_3m.npy"
            ),
            np.array(context_goals),
        )
        self.envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument("-e", "--eval", default=False, action="store_true")
    args = parser.parse_args()

    config = get_config(args.exp_config)
    d = DataCollector(config, args.eval)
    d.train()
