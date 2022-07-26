import argparse
import os
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
    apply_obs_transforms_batch, apply_obs_transforms_obs_space,
    get_active_obs_transforms)
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.behavioral_cloning.agents import (MapStudent,
                                                            WaypointStudent,
                                                            WaypointTeacher)
from habitat_baselines.utils.common import (action_to_velocity_control,
                                            batch_obs)
from habitat_baselines.utils.env_utils import construct_envs

CHECKPOINT_PATH = ""
LAST_TEACHER_BATCH = 1000


def copy_batch(batch, device):
    batch_copy = defaultdict(list)
    for sensor in batch:
        batch_copy[sensor] = batch[sensor].detach().clone()
    return batch_copy


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
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        self.teacher = WaypointTeacher(self.config.RL)
        # self.student = MapStudent(self.config.RL)
        self.student = WaypointStudent(self.config.RL)
        self.obs_transforms = get_active_obs_transforms(self.config)
        # self.student.actor_critic.load_state_dict(
        #      torch.load(CHECKPOINT_PATH)['state_dict']
        # )
        self.batch_length = self.config.BATCH_LENGTH
        self.batch_save_length = self.config.BATCHES_PER_CHECKPOINT

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

        self.optimizer = optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad,
                    self.student.actor_critic.parameters(),
                )
            ),
            lr=self.config.SL_LR,
        )
        batch_num = 0
        spl_deq = deque([0.0], maxlen=100)
        succ_deq = deque([0.0], maxlen=100)
        all_done = 0
        action_loss = 0
        for iteration in range(1, int(5e8)):
            current_episodes = self.envs.current_episodes()

            # in_batch = copy_batch(batch, device=self.device)
            in_batch = batch
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
            (
                _,
                student_actions,
                _,
                student_hidden_states,
            ) = self.student.actor_critic.act(
                in_batch,
                in_hidden,
                in_prev_actions,
                in_not_done,
                deterministic=False,
            )

            # Loss and update
            student_vels = [
                action_to_velocity_control(a, "VELOCITY_CONTROL")
                for a in student_actions.to(device="cpu")
            ]
            teacher_vels = [
                action_to_velocity_control(a, "VELOCITY_CONTROL")
                for a in teacher_actions.to(device="cpu")
            ]
            print(
                "student_vels: ",
                student_actions,
                "LIN VEL :",
                student_vels[0]["action"]["action_args"]["lin_vel"],
                "ANG VEL :",
                student_vels[0]["action"]["action_args"]["ang_vel"],
            )
            print(
                "teacher_vels: ",
                teacher_actions,
                "LIN VEL :",
                teacher_vels[0]["action"]["action_args"]["lin_vel"],
                "ANG VEL :",
                teacher_vels[0]["action"]["action_args"]["ang_vel"],
            )
            # mse_loss = F.mse_loss(
            #     torch.clip(student_actions, min=-1, max=1),
            #     torch.clip(teacher_actions, min=-1, max=1),
            # )
            mse_loss = F.mse_loss(student_actions, teacher_actions)
            print("MSE LOSS: ", mse_loss)

            action_loss += mse_loss
            # student_actions_tanh = torch.tanh(student_actions)
            # teacher_labels = torch.tanh(teacher_actions)
            #
            # action_loss += (
            #     (student_actions_tanh - teacher_labels).pow(2).mean()
            # )

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
                        f"{np.mean(spl_deq):.4f}"
                        ".ckpt"
                    )
                    if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
                        os.makedirs(self.config.CHECKPOINT_FOLDER)
                    torch.save(
                        checkpoint,
                        os.path.join(self.config.CHECKPOINT_FOLDER, file_name),
                    )
                action_loss = 0

            # Step environment
            # teacher_thresh = 1.0 - float(batch_num) / float(LAST_TEACHER_BATCH)
            # teacher_drives = np.random.rand() < teacher_thresh

            ## teacher drives
            if self.config.RL.TEACHER_FORCE:
                teacher_prev_actions.copy_(teacher_actions)
                student_prev_actions.copy_(teacher_actions)
            else:
                ## student always drives
                student_prev_actions.copy_(student_actions)
                teacher_actions = student_actions
                teacher_prev_actions.copy_(teacher_actions)
            step_actions = [
                action_to_velocity_control(a, "VELOCITY_CONTROL")
                for a in teacher_actions.to(device="cpu")
            ]
            outputs = self.envs.step(step_actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for idx, done in enumerate(dones):
                if done:
                    spl_deq.append(infos[idx]["spl"])
                    succ_deq.append(infos[idx]["success"])

            batch = batch_obs(observations, device=self.device)
            # batch = apply_obs_transforms_batch(batch, self.obs_transforms)

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
