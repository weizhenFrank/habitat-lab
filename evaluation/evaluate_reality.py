from collections import OrderedDict, defaultdict

import argparse

import random
import numpy as np
import torch
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from PIL import Image
#from map_and_plan_agent.slam import DepthMapperAndPlanner

import habitat
from habitat.sims import make_sim
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    PointNavResNetPolicy,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer


DEVICE = torch.device("cpu")
SIMULATOR_REALITY_ACTIONS = {0: "stop", 1: "forward", 2: "left", 3: "right"}
LOG_FILENAME = "exp.navigation.log"
MAX_DEPTH = 10.0


class NavEnv:
    def __init__(
        self, forward_step, angle_step, is_blind=False, sensors=["RGB_SENSOR"]
    ):
        config = habitat.get_config()

        log_mesg(
            "env: forward_step: {}, angle_step: {}".format(
                forward_step, angle_step
            )
        )

        config.defrost()
        config.PYROBOT.SENSORS = sensors
        config.PYROBOT.RGB_SENSOR.WIDTH = 256
        config.PYROBOT.RGB_SENSOR.HEIGHT = 256
        config.PYROBOT.DEPTH_SENSOR.WIDTH = 256
        config.PYROBOT.DEPTH_SENSOR.HEIGHT = 256
        config.freeze()

        self._reality = make_sim(id_sim="PyRobot-v0", config=config.PYROBOT)
        self._angle = (angle_step / 180) * np.pi
        self._pointgoal_key = "pointgoal_with_gps_compass"
        self.is_blind = is_blind

        if not is_blind:
            sensors_dict = {
                **self._reality.sensor_suite.observation_spaces.spaces
            }
        else:
            sensors_dict = {}

        sensors_dict[self._pointgoal_key] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = SpaceDict(sensors_dict)

        self.action_space = spaces.Discrete(4)

        self._actions = {
            "forward": [forward_step, 0, 0],
            "left": [0, 0, self._angle],
            "right": [0, 0, -self._angle],
            "stop": [0, 0, 0],
        }

    def _pointgoal(self, agent_state, goal):
        agent_x, agent_y, agent_rotation = agent_state
        agent_coordinates = np.array([agent_x, agent_y])
        rho = np.linalg.norm(agent_coordinates - goal)
        theta = (
            np.arctan2(
                goal[1] - agent_coordinates[1], goal[0] - agent_coordinates[0]
            )
            - agent_rotation
        )
        theta = theta % (2 * np.pi)
        if theta >= np.pi:
            theta = -((2 * np.pi) - theta)
        return rho, theta

    @property
    def pointgoal_key(self):
        return self._pointgoal_key

    def reset(self, goal_location):
        self._goal_location = np.array(goal_location)
        observations = self._reality.reset()

        base_state = self._get_base_state()

        assert np.all(base_state == 0) == True, (
            "Please restart the roslaunch command. "
            "Current base_state is {}".format(base_state)
        )

        observations[self._pointgoal_key] = self._pointgoal(
            base_state, self._goal_location
        )

        return observations

    def _get_base_state(self):
        base_state = self._reality.base.get_state("odom")
        base_state = np.array(base_state, dtype=np.float32)
        log_mesg("base_state: {:.3f} {:.3f} {:.3f}".format(*base_state))
        return base_state

    @property
    def reality(self):
        return self._reality

    def step(self, action):
        if action not in self._actions:
            raise ValueError("Invalid action type: {}".format(action))
        if action == "stop":
            raise NotImplementedError("stop action not implemented")

        observations = self._reality.step(
            "go_to_relative",
            {
                "xyt_position": self._actions[action],
                "use_map": False,
                "close_loop": True,
                "smooth": False,
            },
        )

        base_state = self._get_base_state()

        observations[self._pointgoal_key] = self._pointgoal(
            base_state, self._goal_location
        )

        return observations


def log_mesg(mesg):
    print(mesg)
    with open(LOG_FILENAME, "a") as f:
        f.write(mesg + "\n")


def load_model(
    path,
    observation_space,
    action_space,
    hidden_size,
    normalize_visual_inputs,
    backbone,
    num_recurrent_layers,
    device,
):

    model = PointNavResNetPolicy(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=hidden_size,
        normalize_visual_inputs=normalize_visual_inputs,
        backbone=backbone,
        num_recurrent_layers=num_recurrent_layers
    )

    model.to(device)

    new_model_params = sum(
        [torch.numel(p) for _, p in model.named_parameters()]
    )

    saved_model = torch.load(path, map_location=device)
    saved_model_params = sum(
        [torch.numel(v) for k, v in saved_model["state_dict"].items()]
    )

    print(
        "new_model_params: {}, saved_model_params: {}".format(
            new_model_params, saved_model_params
        )
    )

    saved_model_state_dict = OrderedDict()
    for k, v in saved_model["state_dict"].items():
        new_k = k.replace("actor_critic.", "")
        new_k2 = new_k.replace("net.visual_encoder.final_fc.0.weight", "net.visual_fc.1.weight")
        new_k3 = new_k2.replace("net.visual_encoder.final_fc.0.bias", "net.visual_fc.1.bias")
        saved_model_state_dict[new_k3] = v

    model.load_state_dict(saved_model_state_dict)

    return model


def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sensors", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument(
        "--normalize-visual-inputs", type=int, required=True, choices=[0, 1]
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["resnet50", "se_resneXt50"],
    )
    parser.add_argument("--num-recurrent-layers", type=int, required=True)
    parser.add_argument("--goal", type=str, required=False, default="0.2,0.0")
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--depth-model", type=str, required=False, default="")
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-plan-baseline", action="store_true")
    args = parser.parse_args()

    vtorch = "1.2.0"
    assert torch.__version__ == vtorch, "Please use torch {}".format(vtorch)

    if args.map_plan_baseline is True:
        assert "RGB_SENSOR" in args.sensors and "DEPTH_SENSOR" in args.sensors

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    log_mesg("Starting new episode")

    env = NavEnv(
        forward_step=0.25,
        angle_step=30,
        is_blind=(args.sensors == ""),
        sensors=args.sensors.split(","),
    )
    goal_list = [args.goal_x, args.goal_y]
    goal_location = np.array(goal_list, dtype=np.float32)
    log_mesg("Goal location: {}".format(goal_location))
    device = torch.device("cpu")

    if args.depth_model != "":
        d_model = torch.load(args.depth_model, map_location=device)["model"]
        d_model = d_model.eval()
        print("depth_model:")
        print(d_model)

    sensors_dict = {**env._reality.sensor_suite.observation_spaces.spaces}

    if args.depth_only:
        del sensors_dict["rgb"]
        print("Deleting Sensor from model: rgb")

    sensors_dict[env.pointgoal_key] = spaces.Box(
        low=np.finfo(np.float32).min,
        high=np.finfo(np.float32).max,
        shape=(2,),
        dtype=np.float32,
    )

    num_processes = 1

    if args.map_plan_baseline is False:
        model = load_model(
            path=args.model_path,
            observation_space=SpaceDict(sensors_dict),
            action_space=env.action_space,
            hidden_size=args.hidden_size,
            normalize_visual_inputs=bool(args.normalize_visual_inputs),
            backbone=args.backbone,
            num_recurrent_layers=args.num_recurrent_layers,
            device=device,
        )
        model = model.eval()

        test_recurrent_hidden_states = torch.zeros(
            model.net.num_recurrent_layers,
            num_processes,
            args.hidden_size,
            device=DEVICE,
        )
        test_recurrent_hidden_states = torch.zeros(
            model.net.num_recurrent_layers,
            num_processes,
            args.hidden_size,
            device=DEVICE,
        )
        prev_actions = torch.zeros(num_processes, 1, device=DEVICE)
 #   else:
 #       model = DepthMapperAndPlanner(
 #           map_size_cm=1200,
 #           out_dir=None,
 #           mark_locs=True,
 #           reset_if_drift=True,
 #           count=-1,
 #           close_small_openings=True,
 #           recover_on_collision=True,
 #           fix_thrashing=True,
 #           goal_f=1.1,
 #           point_cnt=2,
 #       )
 #       model.reset()

        old_new_action_mapping = {0: 1, 1: 2, 2: 3, 3: 0}

    not_done_masks = torch.zeros(num_processes, 1, device=DEVICE)

    observations = env.reset(goal_location)

    timestep = -1

    while True:
        timestep += 1
        observations = [observations]

        goal = observations[0][env.pointgoal_key]
        log_mesg(
            "Your goal is to get to: {:.3f}, {:.3f}  "
            "rad ({:.2f} degrees)".format(
                goal[0], goal[1], (goal[1] / np.pi) * 180
            )
        )

        batch = defaultdict(list)

        for obs in observations:
            for sensor in obs:
                batch[sensor].append(to_tensor(obs[sensor]))

        for sensor in batch:
            batch[sensor] = torch.stack(batch[sensor], dim=0).to(
                device=DEVICE, dtype=torch.float
            )

        if args.depth_model != "":
            with torch.no_grad():
                rgb_stretch = batch["rgb"].permute(0, 3, 1, 2) / 255.0

                # FASTDEPTH expects a NCHW order
                depth_stretch = d_model(rgb_stretch)
                depth_stretch = torch.clamp(depth_stretch / MAX_DEPTH, 0, 1.0)
                batch["depth"] = depth_stretch.permute(0, 2, 3, 1)

            # torch.save(batch, "episode/timestep_{}.pt".format(timestep))

        if args.map_plan_baseline is False:
            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = model.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
            prev_actions.copy_(actions)
        else:
            assert "rgb" in batch
            assert "depth" in batch
            assert batch["rgb"].shape[0] == 1

            slam_batch_input = {}
            slam_batch_input["rgb"] = batch["rgb"].numpy()[0]
            slam_batch_input["depth"] = batch["depth"].numpy()[0]
            slam_batch_input["pointgoal"] = batch[
                "pointgoal_with_gps_compass"
            ].numpy()[0]

            slam_action = model.act(slam_batch_input)
            actions = torch.Tensor(
                [old_new_action_mapping[slam_action]]
            ).unsqueeze(0)

        simulation_action = actions[0].item()
        reality_action = SIMULATOR_REALITY_ACTIONS[simulation_action]
        print("reality_action:", reality_action)
        # input("Press key to continue")
        if reality_action != "stop":
            observations = env.step(reality_action)
            not_done_masks = torch.ones(num_processes, 1, device=DEVICE)
        else:
            print("STOP called, episode over.")
            print("Distance to goal: {:.3f}m".format(goal[0]))
            return


if __name__ == "__main__":
    main()
