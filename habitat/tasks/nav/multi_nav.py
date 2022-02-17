from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import (
    IntegratedPointGoalGPSAndCompassSensor,
    NavigationTask,
    TopDownMap,
)
from habitat.utils.visualizations import maps

from .spot_utils.daisy_env import *
from .spot_utils.quadruped_env import *


@registry.register_task(name="MultiNav-v0")
class MultiNavigationTask(NavigationTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.robot_id = None
        self.robots = self._config.ROBOTS
        self.robot_files = self._config.ROBOT_URDFS
        # if task reset happens everytime episode is reset, then create previous state deck here
        ## init prev states
        self.prev_states = deque(maxlen=self._config.Z.PREV_WINDOW)
        self.prev_actions = deque(maxlen=self._config.Z.PREV_WINDOW)

    def reset(self, episode: Episode):
        # If robot was never spawned or was removed with previous scene
        rand_robot = np.random.randint(0, len(self.robot_files))
        # if randomly selected robot is not the current robot already spawned
        # if there is a robot/ URDF created
        if (
            self.robot_id is not None
            and self.robot_id.object_id != -1
            and self.robot_wrapper.id != rand_robot
        ):
            self.art_obj_mgr.remove_object_by_id(self.robot_id.object_id)
            self.robot_id = None
        if self.robot_id is None or self.robot_id.object_id == -1:
            self._load_robot(rand_robot)

        observations = super().reset(episode)
        observations["robot_id"] = rand_robot
        observations["urdf_params"] = self.robot_wrapper.urdf_params

        self.default_state_shape = 2  # for rho, phi
        default_state = np.zeros(self.default_state_shape)
        default_states = [default_state] * self._config.Z.PREV_WINDOW
        self.prev_states = deque(
            default_states, maxlen=self._config.Z.PREV_WINDOW
        )

        self.default_action_shape = 3  # for vx, vy, vt
        default_action = np.zeros(self.default_action_shape)
        default_actions = [default_action] * self._config.Z.PREV_WINDOW
        self.prev_actions = deque(
            default_actions, maxlen=self._config.Z.PREV_WINDOW
        )

        observations["prev_states"] = self.prev_states
        observations["prev_actions"] = self.prev_actions
        return observations

    def _load_robot(self, rand_robot):
        # Add robot into the simulator

        self.art_obj_mgr = self._sim.get_articulated_object_manager()
        self.robot_id = self.art_obj_mgr.add_articulated_object_from_urdf(
            self.robot_files[rand_robot], fixed_base=False
        )
        # obj_mgr = self._sim.get_object_template_manager()
        # self.cube_id = self._sim.add_object_by_handle(obj_mgr.get_template_handles("cube")[0])

        if self.robot_id.object_id == -1:
            raise ValueError("Could not load " + robot_file)

        # Initialize robot wrapper
        self.robot_wrapper = eval(self.robots[rand_robot])(
            sim=self._sim, robot=self.robot_id, rand_id=rand_robot
        )
        self.robot_wrapper.id = rand_robot
        if self.robot_wrapper.name != "Locobot":
            self.robot_id.joint_positions = (
                self.robot_wrapper._initial_joint_positions
            )

        # depth_sensor = self._sim._sensors["depth"]
        # depth_pos_offset = np.array([0.0, 0.0, 0.0]) + self.robot_wrapper.camera_spawn_offset
        # depth_sensor._spec.position = depth_pos_offset
        # depth_sensor._sensor_object.set_transformation_from_spec()

    def step(self, action: Dict[str, Any], episode: Episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        task_action = self.actions[action_name]
        observations = task_action.step(**action["action_args"], task=self)
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )
        observations["robot_id"] = self.robot_wrapper.id
        observations["urdf_params"] = self.robot_wrapper.urdf_params

        prev_states_array = np.array(self.prev_states)
        # assert len(prev_states_array) == self.default_state_shape * len(self.prev_states)
        observations["prev_states"] = prev_states_array

        current_state = observations[
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
        ]
        self.prev_states.append(current_state)

        prev_actions_array = np.array(self.prev_actions)
        # assert len(prev_actions_array) == self.default_action_shape * len(self.prev_actions)
        observations["prev_actions"] = prev_actions_array

        current_action = np.array(
            [
                action["action_args"]["lin_vel"],
                action["action_args"]["hor_vel"],
                action["action_args"]["ang_vel"],
            ]
        )
        self.prev_actions.append(current_action)
        # observations['z_model'] = self.robot_wrapper.z_model

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        return observations