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
    MapPretrainedStudent,
    MapStudent,
    WaypointStudent,
    WaypointTeacher,
)
from habitat_baselines.utils.common import (
    action_to_velocity_control,
    batch_obs,
)
from habitat_baselines.utils.env_utils import construct_envs
from torch.utils.tensorboard import SummaryWriter

CHECKPOINT_PATH = ""
LAST_TEACHER_BATCH = 1000


@baseline_registry.register_trainer(name="behavioral_cloning")
class BehavioralCloning(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        logger.info(f"env config: {config}")

        # Faster loading
        # config.defrost()
        # config.TASK_CONFIG.DATASET.SPLIT = 'val'
        # config.freeze()

        self.config = config
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        self.teacher = WaypointTeacher(self.config.RL)
        self.student = MapPretrainedStudent(self.config)

        self.obs_transforms = get_active_obs_transforms(self.config)
        self.batch_length = self.config.BATCH_LENGTH
        self.batch_save_length = self.config.BATCHES_PER_CHECKPOINT
        self.clip_mse = config.get("CLIP_MSE", False)
        self.loss = config.get("LOSS", "log_prob")
        self.regress = config.get("REGRESS", "actions")
        self.debug_waypoint = config.get("DEBUG_WAYPOINT", False)

        self.mse_weight = config.get("MSE_WEIGHT", 1)
        self.is_weight = config.get("IS_WEIGHT", 1)

        self.tb_dir = self.config.TENSORBOARD_DIR
        print("USING LOSS: ", self.loss)

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
        # Student tensors
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

        self.teacher.actor_critic.eval()

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
        spl_deq = deque([0.0], maxlen=100)
        succ_deq = deque([0.0], maxlen=100)
        action_loss = 0

        if self.tb_dir != "":
            print(f"Creating tensorboard at {self.tb_dir}...")
            os.makedirs(self.tb_dir, exist_ok=True)
            writer = SummaryWriter(self.tb_dir)
        else:
            writer = None
        for iteration in range(1, 1000000):
            in_hidden = student_hidden_states.detach().clone()
            in_prev_actions = student_prev_actions.detach().clone()
            in_not_done = not_done_masks.clone()
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
            if self.debug_waypoint:
                batch["context_waypoint"] *= 0

            (
                _,
                student_actions,
                _,
                student_hidden_states,
            ) = self.student.actor_critic.act(
                batch,
                in_hidden,
                in_prev_actions,
                in_not_done,
                deterministic=False,
            )
            if self.regress == "waypoint_rma_2":
                ## make student map encoder output same as waypoint

                teacher_label = torch.stack(
                    [
                        batch["context_waypoint"][:, 0],
                        torch.cos(-batch["context_waypoint"][:, 1]),
                        torch.sin(-batch["context_waypoint"][:, 1]),
                    ],
                    -1,
                )
                assert (
                    self.student.actor_critic.net.map_ce.shape
                    == teacher_label.shape
                )

                loss = F.mse_loss(
                    self.student.actor_critic.net.map_ce,
                    teacher_label,
                )
            elif self.regress == "actions":
                if self.loss == "mse":
                    assert student_actions.shape == teacher_actions.shape

                    loss = F.mse_loss(student_actions, teacher_actions)
                elif self.loss == "log_prob":
                    loss = -self.student.actor_critic.distribution.log_prob(
                        teacher_actions
                    ).mean()
            action_loss += loss
            del loss

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
                    f"  avg_succ: {np.mean(succ_deq):.4f}"
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
                        f"{batch_num:03d}_"
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
                # Update tensorboard
                if writer is not None:
                    metrics_data = {
                        "ep_success": np.mean(succ_deq),
                        "ep_spl": np.mean(spl_deq),
                    }
                    loss_data = {"action_loss": action_loss.item()}
                    writer.add_scalars("metrics", metrics_data, batch_num)
                    writer.add_scalars("loss", loss_data, batch_num)
                action_loss = 0
            # Step environment
            # teacher_thresh = 1.0 - float(batch_num) / float(LAST_TEACHER_BATCH)
            # teacher_drives = np.random.rand() < teacher_thresh

            ## teacher drives
            if self.config.RL.TEACHER_FORCE:
                execute_actions = teacher_actions
            else:
                execute_actions = student_actions
            student_prev_actions.copy_(execute_actions)
            teacher_prev_actions.copy_(execute_actions)
            step_actions = [
                action_to_velocity_control(a, "VELOCITY_CONTROL")
                for a in execute_actions.to(device="cpu")
            ]
            outputs = self.envs.step(step_actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for idx, done in enumerate(dones):
                if done:
                    spl_deq.append(infos[idx]["spl"])
                    succ_deq.append(infos[idx]["success"])

            if "rma" in self.regress:
                del self.student.actor_critic.net.map_ce
                del batch

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
    d = BehavioralCloning(config)
    d.train()
