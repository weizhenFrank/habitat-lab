import argparse
import os
import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.behavioral_cloning.agents import (
    BaselineStudent,
    MapStudent,
)
from habitat_baselines.utils.common import (
    action_to_velocity_control,
    batch_obs,
)
from habitat_baselines.utils.env_utils import construct_envs
from torch.utils.tensorboard import SummaryWriter

CHECKPOINT_PATH = ""
LAST_TEACHER_BATCH = 1000


@baseline_registry.register_trainer(name="behavioral_cloning_planner")
class BehavioralCloningPlanner(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        logger.info(f"env config: {config}")

        self.config = config
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        self.student = MapStudent(self.config)
        self.obs_transforms = get_active_obs_transforms(self.config)
        self.batch_length = self.config.BATCH_LENGTH
        self.batch_save_length = self.config.BATCHES_PER_CHECKPOINT

        self.tb_dir = self.config.TENSORBOARD_DIR

    def copy_batch(self, batch):
        batch_copy = defaultdict(list)
        for sensor in batch:
            batch_copy[sensor] = batch[sensor].detach().clone()
        return batch_copy

    def train(self):
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        observations = self.envs.reset()

        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # Student tensors
        num_actions = 2
        student_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.student.actor_critic.net.num_recurrent_layers,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )
        student_prev_actions = torch.zeros(
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

        for n, p in self.student.actor_critic.named_parameters():
            if p.requires_grad:
                print("requires grad: ", n)
        self.optimizer = optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad,
                    self.student.actor_critic.parameters(),
                )
            ),
            lr=self.config.SL_LR,
            weight_decay=self.config.SL_WD,
        )
        batch_num = 0
        spl_deq = deque([0.0], maxlen=50)
        succ_deq = deque([0.0], maxlen=50)
        action_loss = 0

        if self.tb_dir != "":
            print(f"Creating tensorboard at {self.tb_dir}...")
            os.makedirs(self.tb_dir, exist_ok=True)
            writer = SummaryWriter(self.tb_dir)
        else:
            writer = None
        for iteration in range(1, 1000000):
            (
                _,
                student_actions,
                _,
                student_hidden_states,
            ) = self.student.actor_critic.act(
                self.copy_batch(batch),
                student_hidden_states.detach().clone(),
                student_prev_actions.detach().clone(),
                not_done_masks.detach().clone(),
                deterministic=False,
            )
            print("student_actions: ", student_actions)

            batch["context_waypoint"] = torch.squeeze(
                batch["context_waypoint"], dim=1
            ).detach()

            teacher_label = torch.stack(
                [
                    batch["context_waypoint"][:, 0],
                    torch.sin(batch["context_waypoint"][:, 1]),
                    torch.cos(batch["context_waypoint"][:, 1]),
                ],
                -1,
            )

            assert (
                self.student.actor_critic.net.map_ce.shape
                == teacher_label.shape
            )

            # print("student: ", self.student.actor_critic.net.map_ce)
            # print("teacher: ", teacher_label)
            print("mapce1 : ", self.student.actor_critic.net.map_ce)
            # teacher_label = torch.zeros_like(student_actions)
            # action_loss += F.mse_loss(
            #     student_actions,
            #     teacher_label,
            # )
            action_loss += F.mse_loss(
                self.student.actor_critic.net.map_ce[:, 0],
                # teacher_label[:, 0],
                torch.zeros_like(self.student.actor_critic.net.map_ce[:, 0]),
            )
            # loss_theta = F.mse_loss(
            #     self.a[:, 1:],
            #     teacher_label[:, 1:],
            # )

            # loss_r = F.mse_loss(
            #     self.student.actor_critic.net.map_ce[:, 0],
            #     teacher_label[:, 0],
            # )
            #
            # loss_theta = F.mse_loss(
            #     self.student.actor_critic.net.map_ce[:, 1:],
            #     teacher_label[:, 1:],
            # )

            del teacher_label
            # action_loss += loss_r + 4 * loss_theta
            # del loss_r
            # del loss_theta

            if iteration % self.batch_length == 0:
                self.optimizer.zero_grad()

                action_loss /= float(self.batch_length)
                action_loss.backward()
                self.optimizer.step()

                batch_num += 1

                print(
                    f"#{batch_num}"
                    f"  action_loss: {action_loss.item():.4f}"
                    f"  avg_spl: {np.mean(spl_deq):.4f}"
                    f"  avg_succ: {np.mean(succ_deq):.4f}, {len(succ_deq)}"
                )

                if batch_num % self.batch_save_length == 0:
                    checkpoint = {
                        "state_dict": {
                            "actor_critic." + k: v
                            for k, v in self.student.actor_critic.state_dict().items()
                        },
                        "config": self.config,
                    }
                    file_name = (
                        f"ckpt_{batch_num:03d}_"
                        f"{action_loss.item():.4f}_"
                        f"{np.mean(spl_deq):.4f}_"
                        f"{np.mean(succ_deq):.4f}"
                        ".pth"
                    )
                    if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
                        os.makedirs(self.config.CHECKPOINT_FOLDER)
                    torch.save(
                        checkpoint,
                        os.path.join(self.config.CHECKPOINT_FOLDER, file_name),
                    )

                    del checkpoint
                # Update tensorboard
                if writer is not None:
                    metrics_data = {
                        "ep_success": np.mean(succ_deq),
                        "ep_spl": np.mean(spl_deq),
                    }
                    loss_data = {"action_loss": action_loss.item()}
                    writer.add_scalars("metrics", metrics_data, batch_num)
                    writer.add_scalars("loss", loss_data, batch_num)
                del action_loss
                action_loss = 0
            # Step environment
            # teacher_thresh = 1.0 - float(batch_num) / float(LAST_TEACHER_BATCH)
            # teacher_drives = np.random.rand() < teacher_thresh

            ## teacher drives
            student_prev_actions.copy_(student_actions)
            step_actions = [
                action_to_velocity_control(a, "VELOCITY_CONTROL")
                for a in student_actions.detach().cpu().unbind(0)
            ]
            outputs = self.envs.step(step_actions)
            del step_actions
            del observations
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            del outputs
            del batch
            for idx, done in enumerate(dones):
                if done:
                    spl_deq.append(infos[idx]["spl"])
                    succ_deq.append(infos[idx]["success"])

            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.device,
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
    args = parser.parse_args()

    config = get_config(args.exp_config)
    d = BehavioralCloningPlanner(config)
    d.train()
