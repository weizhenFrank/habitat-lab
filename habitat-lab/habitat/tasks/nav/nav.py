#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO, lots of typing errors in here

from typing import Any, List, Optional, Tuple

import attr
import magnum as mn
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
    get_rpy
)
from habitat.utils.visualizations import fog_of_war, maps
from habitat_sim.bindings import RigidState
from habitat_sim.physics import VelocityControl
from habitat_sim.physics import MotionType
import habitat_sim

from .spot_utils.quadruped_env import A1, AlienGo, Laikago, Spot
from .spot_utils.daisy_env import Daisy, Daisy_4legged
from .spot_utils.raibert_controller import Raibert_controller
from .spot_utils.raibert_controller import Raibert_controller_turn_stable

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass
cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 128


def merge_sim_episode_config(sim_config: Config, episode: Episode) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius."""

    room_id: str = attr.ib(default=None, validator=not_none_validator)
    room_name: Optional[str] = None


@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None


@registry.register_sensor
class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def get_observation(
        self,
        observations,
        episode: NavigationEpisode,
        *args: Any,
        **kwargs: Any,
    ):
        source_position = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            source_position, rotation_world_start, goal_position
        )


@registry.register_sensor
class ImageGoalSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoal"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        # to be sure that the rotation is the same for the same episode_id
        # since the task is currently using pointnav Dataset.
        seed = abs(hash(episode.episode_id)) % (2 ** 32)
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal_with_gps_compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "heading"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        return self._quat_to_xy_heading(rotation_world_agent.inverse())


@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        return self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )


@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "gps"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-agent_position[2], agent_position[0]], dtype=np.float32
            )
        else:
            return agent_position.astype(np.float32)


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "proximity"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        return np.array(
            [
                self._sim.distance_to_closest_obstacle(
                    current_position, self._max_detection_radius
                )
            ],
            dtype=np.float32,
        )

@registry.register_measure
class Proximity(Measure):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "proximity"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = 100.0

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position

        self._metric = self._sim.distance_to_closest_obstacle(
                        current_position, self._max_detection_radius
                        )

@registry.register_measure
class Success(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class SoftSPL(SPL):
    r"""Soft SPL

    Similar to SPL with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "softspl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class Collisions(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "collisions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if self._metric is None:
            self._metric = {"count": 0, "is_collision": False}
        self._metric["is_collision"] = False
        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True

@registry.register_measure
class NumActions(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "num_actions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._metric += 1

@registry.register_measure
class EpisodeDistance(Measure):
    r"""Success weighted by Completion Time

    """
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "episode_distance"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._metric = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        pass

@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure"""

    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count: Optional[int] = None
        self._map_resolution = config.MAP_RESOLUTION
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: Optional[Tuple[int, int]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.DRAW_BORDER,
        )

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.DRAW_VIEW_POINTS:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        if goal.view_points is not None:
                            for view_point in goal.view_points:
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                    except AttributeError:
                        pass

    def _draw_goals_positions(self, episode):
        if self._config.DRAW_GOAL_POSITIONS:

            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        self._draw_point(
                            goal.position, maps.MAP_TARGET_POINT_INDICATOR
                        )
                    except AttributeError:
                        pass

    def _draw_goals_aabb(self, episode):
        if self._config.DRAW_GOAL_AABBS:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                        if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            self._top_down_map.shape[0:2],
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.DRAW_SHORTEST_PATH:
            _shortest_path_points = (
                self._sim.get_straight_shortest_path_points(
                    agent_position, episode.goals[0].position
                )
            )
            self._shortest_path_points = [
                maps.to_grid(
                    p[2], p[0], self._top_down_map.shape[0:2], sim=self._sim
                )
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=2.0
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height < height < ref_floor_height + ceiling_height

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target parts last to avoid overlap
        self._draw_goals_view_points(episode)
        self._draw_goals_aabb(episode)
        self._draw_goals_positions(episode)

        self._draw_shortest_path(episode, agent_position)

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = self.line_thickness
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )


@registry.register_measure
class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.DISTANCE_TO == "POINT":
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in episode.goals],
                    episode,
                )
            elif self._config.DISTANCE_TO == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
                )

            self._previous_position = current_position
            self._metric = distance_to_target


@registry.register_task_action
class MoveForwardAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.MOVE_FORWARD)


@registry.register_task_action
class TurnLeftAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.TURN_LEFT)


@registry.register_task_action
class TurnRightAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.TURN_RIGHT)


@registry.register_task_action
class StopAction(SimulatorTaskAction):
    name: str = "STOP"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_stop_called = True  # type: ignore
        return self._sim.get_observations_at()  # type: ignore


@registry.register_task_action
class LookUpAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.LOOK_UP)


@registry.register_task_action
class LookDownAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.LOOK_DOWN)


@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    # TODO @maksymets: Propagate through Simulator class
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "TELEPORT"

    def step(
        self,
        *args: Any,
        position: List[float],
        rotation: List[float],
        **kwargs: Any,
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            return self._sim.get_observations_at()  # type: ignore

        return self._sim.get_observations_at(
            position=position, rotation=rotation, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([self.COORDINATE_MIN] * 3),
                    high=np.array([self.COORDINATE_MAX] * 3),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )


@registry.register_task_action
class VelocityAction(SimulatorTaskAction):
    name: str = "VELOCITY_CONTROL"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        config = kwargs["config"]
        self.config = config
        self.min_lin_vel, self.max_lin_vel = config.LIN_VEL_RANGE
        self.min_strafe_vel, self.max_strafe_vel = config.STRAFE_VEL_RANGE
        self.min_ang_vel, self.max_ang_vel = config.ANG_VEL_RANGE
        self.min_abs_lin_speed = config.MIN_ABS_LIN_SPEED
        self.min_abs_strafe_speed = config.MIN_ABS_STRAFE_SPEED
        self.min_abs_ang_speed = config.MIN_ABS_ANG_SPEED
        self.time_step = config.TIME_STEP
        self.use_strafe_vel = config.USE_STRAFE
        self.auto_stop = config.AUTO_STOP
        self.max_collisions = config.MAX_COLLISIONS
        self.success_distance = config.SUCCESS_DISTANCE
        self.oblong_robot = config.USE_OBLONG_ROBOT
        self.urdf_robot = config.USE_URDF_ROBOT
        self.counter = 0

        self.robot_name = config.ROBOT
        self.ac_freq_ratio = config.AC_FREQ_RATIO
        self.ctrl_freq = config.CTRL_FREQ
        self.dt = 1/self.ctrl_freq
        self.pos_gain = np.ones((3,)) * 0.1 # 0.2
        self.vel_gain = np.ones((3,)) * 1.0 # 1.5
        self.pos_gain[2] = 0.1 # 0.7
        self.vel_gain[2] = 1.0 # 1.5
        self.robot_id = None
        self.fixed_base = False
        self.time_per_step = config.TIME_PER_STEP
        self.follow_robot = True
        self.start_position = None

        self._load_robot()
        print('Finished loading robot')
        self.init_state = self.robot_wrapper.calc_state()

        # SET THE CORRECT INITIAL STATE????
        self.raibert_controller.set_init_state(self.init_state)
        
        

    @property
    def action_space(self):
        if self.use_strafe_vel:
            return spaces.Box(
                low=np.array([self.min_lin_vel, self.min_strafe_vel, self.min_ang_vel]),
                high=np.array([self.max_lin_vel, self.max_strafe_vel, self.max_ang_vel]),
                shape=(3,),
                dtype=np.float,
            )
        else:
            return spaces.Box(
                low=np.array([self.min_lin_vel, self.min_ang_vel]),
                high=np.array([self.max_lin_vel, self.max_ang_vel]),
                shape=(2,),
                dtype=np.float,
            )

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
       
        self.start_position = kwargs['episode'].start_position

        print('Rest 0')
        #self.set_robot_pos(kwargs['episode'].start_position)
        print('Reset 1')
        #self.set_robot_rot(kwargs['episode'].start_rotation)
        self.set_init_state(kwargs['episode'].start_position,kwargs['episode'].start_rotation)
        print('Reset 2')
        task.is_stop_called = False  # type: ignore
        # print(self.robot_hab.transformation.translation)

    def set_init_state(self, pos, rot):
        rotation = mn.Quaternion.identity_init()
        print('rotation0')
        rotation.vector = mn.Vector3(rot[1], rot[2], rot[3])
        print('rotation1')
        rotation.scalar = rot[0]
        print('rotation2')
        print(rotation.axis())
        print(rotation.angle())
        transform = mn.Matrix4.rotation(rotation.angle(), rotation.axis())
        print('rotation3')
        transform.translation = mn.Vector3(pos[0], pos[1], pos[2])
        print('rotation4')

        print(transform)

        self.robot_hab.transformation = transform

    def set_robot_pos(self, set_pos):
        """
        - set_pos: 2D coordinates of where the robot will be placed. The height
          will be same as current position.
        """
        base_transform = self.robot_hab.transformation
        pos = base_transform.translation
        base_transform.translation = mn.Vector3(set_pos[0], set_pos[1], set_pos[2])
        # self._sim.set_articulated_object_root_state(self.robot_id, base_transform)
        self.robot_hab.translation = base_transform.translation

    def set_robot_rot(self, rot_quat):
        """
        Set the rotation of the robot along the y-axis. The position will
        remain the same.
        """
        cur_trans = self.robot_hab.transformation

        
        rot = mn.Quaternion().identity_init()

        rot.vector = mn.Vector3()
        self.robot_hab.transformation.rotation = rot

        # rot_trans = mn.Matrix4.rotation(mn.Rad(-1.56), mn.Vector3(1.0, 0, 0))
        # add_rot_mat = mn.Matrix4.rotation(mn.Rad(rot_rad), mn.Vector3(0.0, 0, 1))
        # new_trans = rot_trans @ add_rot_mat
        # new_trans.translation = pos
        # self.robot_hab.transformation.rotation = mn.Quaternion.from_matrix(new_trans.rotation()) 
        # self._sim.set_articulated_object_root_state(self.robot_id, new_trans)

    def step(
        self,
        *args: Any,
        task: EmbodiedTask,
        linear_velocity: float,
        strafe_velocity: float,
        angular_velocity: float,
        time_step: Optional[float] = None,
        allow_sliding: Optional[bool] = None,
        **kwargs: Any,
    ):
        r"""Moves the agent with a provided linear and angular velocity for the
        provided amount of time

        Args:
            linear_velocity: between [-1,1], scaled according to
                             config.LIN_VEL_RANGE
            angular_velocity: between [-1,1], scaled according to
                             config.ANG_VEL_RANGE
            time_step: amount of time to move the agent for
            allow_sliding: whether the agent will slide on collision
        """
        
        if allow_sliding is None:
            allow_sliding = self._sim.config.sim_cfg.allow_sliding  # type: ignore
        if time_step is None:
            time_step = self.time_step
        # Convert from [-1, 1] to [0, 1] range
        linear_velocity = (linear_velocity + 1.0) / 2.0
        strafe_velocity = (strafe_velocity + 1.0) / 2.0
        angular_velocity = (angular_velocity + 1.0) / 2.0
        
        # Scale actions
        linear_velocity = self.min_lin_vel + linear_velocity * (
            self.max_lin_vel - self.min_lin_vel
        )
        strafe_velocity = self.min_strafe_vel + strafe_velocity * (
            self.max_strafe_vel - self.min_strafe_vel
        )
        angular_velocity = self.min_ang_vel + angular_velocity * (
            self.max_ang_vel - self.min_ang_vel
        )

        # Stop is called if both linear/angular speed are below their threshold
        if not self.auto_stop:
            if (
                abs(linear_velocity) < self.min_abs_lin_speed
                and abs(strafe_velocity) < self.min_abs_strafe_speed
                and abs(angular_velocity) < self.min_abs_ang_speed
            ):
                task.is_stop_called = True  # type: ignore
                return self._sim.get_observations_at(position=None, rotation=None)
        else:
            distance_to_target = task.measurements.measures['distance_to_goal'].get_metric()
            if distance_to_target < self.success_distance:
                task.is_stop_called = True    
                return self._sim.get_observations_at(position=None, rotation=None)

        if self.max_collisions != -1:
            collisions_measure = task.measurements.measures['collisions'].get_metric()
            if collisions_measure is not None:
                if collisions_measure['count'] > self.max_collisions:
                    task.is_stop_called = True
                    return self._sim.get_observations_at(position=None, rotation=None)

        angular_velocity = np.deg2rad(angular_velocity)
        self.vel_control.linear_velocity = np.array(
            [strafe_velocity, 0.0, -linear_velocity]
        )
        self.vel_control.angular_velocity = np.array(
            [0.0, angular_velocity, 0.0]
        )
        
        if not self.urdf_robot:
            agent_state = self._sim.get_agent_state()

            # Convert from np.quaternion to mn.Quaternion
            normalized_quaternion = agent_state.rotation
            agent_mn_quat = mn.Quaternion(
                normalized_quaternion.imag, normalized_quaternion.real
            )
            current_rigid_state = RigidState(
                agent_mn_quat,
                agent_state.position,
            )

            # manually integrate the rigid state
            goal_rigid_state = self.vel_control.integrate_transform(
                time_step, current_rigid_state
            )

            # snap rigid state to navmesh and set state to object/agent
            if allow_sliding:
                step_fn = self._sim.pathfinder.try_step  # type: ignore
            else:
                step_fn = self._sim.pathfinder.try_step_no_sliding  # type: ignore

            final_position = step_fn(
                agent_state.position, goal_rigid_state.translation
            )
            final_rotation = [
                *goal_rigid_state.rotation.vector,
                goal_rigid_state.rotation.scalar,
            ]

            # Check if a collision occured
            dist_moved_before_filter = (
                goal_rigid_state.translation - agent_state.position
            ).dot()
            dist_moved_after_filter = (final_position - agent_state.position).dot()

            # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
            # collision _didn't_ happen. One such case is going up stairs.  Instead,
            # we check to see if the the amount moved after the application of the
            # filter is _less_ than the amount moved before the application of the
            # filter.
            EPS = 1e-5
            collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

            agent_observations = self._sim.get_observations_at(
                position=final_position,
                rotation=final_rotation,
                keep_agent_at_new_pose=True,
            )

            # TODO: Make a better way to flag collisions
            self._sim._prev_sim_obs["collided"] = collided  # type: ignore

            return agent_observations
        else:
            

            action = [linear_velocity, strafe_velocity, angular_velocity]

            
            # self.counter = self.counter + 1

            # agent_observations = self._sim.step(action)


            # img = agent_observations['rgb']
            # # print(type(img))
            # # cv2.imwrite('/srv/share3/mrudolph8/test_imgs/imgs/rgb_img_' + str(self.counter) + '.jpg', (img.squeeze() * 255).astype(dtype=np.uint8))
            # if np.any(np.isnan(img)):
            #     print('HAS NANSSSS')
                    #print('SPOT SIM STEP')
            state = self.robot_wrapper.calc_state()
            
            #print('SPOT SIM STEP 0')
            target_speed = np.array([action[0], action[1]]) 
            target_ang_vel = action[2] 
            #print('SPOT SIM STEP 01')
            latent_action = self.raibert_controller.plan_latent_action(state, target_speed, target_ang_vel=target_ang_vel)
            #print('SPOT SIM STEP 02')
            #print('SPOT SIM STEP 0')
            self.raibert_controller.update_latent_action(state, latent_action)
            
            #print('SPOT SIM STEP 1')
            for i in range(self.time_per_step):
                #print('SPOT SIM STEP 2')
                state = self.robot_wrapper.calc_state()
                #print('SPOT SIM STEP 3')
                raibert_action = self.raibert_controller.get_action(state, i+1)
                #print('SPOT SIM STEP 4')
                self.robot_wrapper.apply_robot_action(raibert_action, self.pos_gain, self.vel_gain)
                #print('SPOT SIM STEP 5')
                
                #print('SPOT SIM STEP 6')
                self._sim.step_physics(self.dt)
            #print('SPOT SIM STEP 7')
            if self.follow_robot:
                self._follow_robot()
            sim_obs = self._sim.get_sensor_observations(0)
            agent_observations = self._sim._sensor_suite.get_observations(sim_obs)
            # img = agent_observations['rgb']
            
            # #print(type(img))
            # img = (img.squeeze() * 255).astype(dtype=np.uint8)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, str(self.robot_hab.transformation.translation), (20, 20), font, 0.5, (0, 0, 0), 2)
            # cv2.imwrite('/srv/share3/mrudolph8/test_imgs/imgs/rgb_img_'  + str(int(10*self.start_position[0])) + '_' + str(self.counter) + '.jpg', img)
            #print('SPOT SIM 8')

            self.counter += 1
            
            return agent_observations

    def _load_robot(self):
        print('loading robot')
        if not self.config.get('LOAD_ROBOT', True):
            return

        if self.robot_id is None:
            agent_config = self.config
            robot_file = agent_config.ROBOT_URDF
            art_obj_mgr = self._sim.get_articulated_object_manager()
            # backend_cfg = habitat_sim.SimulatorConfiguration()
            self.robot_hab = art_obj_mgr.add_articulated_object_from_urdf(robot_file, fixed_base=self.fixed_base)
            self.robot_id = self.robot_hab.object_id
            if self.robot_id == -1:
                raise ValueError('Could not load ' + robot_file)

        jms = []
        jms.append(habitat_sim.physics.JointMotorSettings(
                0,  # position_target
                self.pos_gain[0],  # position_gain
                0,  # velocity_target
                self.vel_gain[0],  # velocity_gain
                10.0,  # max_impulse
            ))

        jms.append(habitat_sim.physics.JointMotorSettings(
                        0,  # position_target
                        self.pos_gain[1],  # position_gain
                        0,  # velocity_target
                        self.vel_gain[1],  # velocity_gain
                        10.0,  # max_impulse
                    ))
        jms.append(habitat_sim.physics.JointMotorSettings(
                        0,  # position_target
                        self.pos_gain[2],  # position_gain
                        0,  # velocity_target
                        self.vel_gain[2],  # velocity_gain
                        10.0,  # max_impulse
                    ))      
        for i in range(12):
            self.robot_hab.update_joint_motor(i, jms[np.mod(i,3)])
        rot_trans = mn.Matrix4.rotation(mn.Rad(-1.56), mn.Vector3(1.0, 0, 0))

        if self.robot_name == 'A1':
            self.robot_wrapper = A1(sim=self._sim, robot=self.robot_hab, robot_id=self.robot_id, dt=self.dt)
        elif self.robot_name == 'AlienGo':
            self.robot_wrapper = AlienGo(sim=self._sim, robot=self.robot_hab, robot_id=self.robot_id, dt=self.dt)
        elif self.robot_name == 'Daisy':
            self.robot_wrapper = Daisy(sim=self._sim, robot=self.robot_hab, robot_id=self.robot_id, dt=self.dt)
        elif self.robot_name == 'Laikago':
            self.robot_wrapper = Laikago(sim=self._sim, robot=self.robot_hab, robot_id=self.robot_id, dt=self.dt)
        elif self.robot_name == 'Daisy_4legged':
            self.robot_wrapper = Daisy4(sim=self._sim, robot=self.robot_hab, robot_id=self.robot_id, dt=self.dt)
        elif self.robot_name == 'Spot':
            self.robot_wrapper = Spot(sim=self._sim, robot=self.robot_hab, robot_id=self.robot_id, dt=self.dt)

        self.robot_wrapper.robot_specific_reset()
        
        # Set up Raibert controller and link it to spot
        action_limit = np.zeros((12, 2))
        action_limit[:, 0] = np.zeros(12) + np.pi / 2
        action_limit[:, 1] = np.zeros(12) - np.pi / 2
        self.raibert_controller = Raibert_controller_turn_stable(control_frequency=self.ctrl_freq, num_timestep_per_HL_action=self.time_per_step, robot=self.robot_name)
        print('FINISHED LOADING ROBOT')

    def _follow_robot(self):
        #robot_state = self.sim.get_articulated_object_root_state(self.robot_id)
        robot_state = self.robot_hab.transformation
        node = self._sim._default_agent.scene_node
        self.h_offset = 0.69
        cam_pos = mn.Vector3(0, 0.0, 0)

        look_at = mn.Vector3(1, 0.0, 0)
        look_at = robot_state.transform_point(look_at)

        cam_pos = robot_state.transform_point(cam_pos)

        node.transformation = mn.Matrix4.look_at(
                cam_pos,
                look_at,
                mn.Vector3(0, 1, 0))

        self.cam_trans = node.transformation
        self.cam_look_at = look_at
        self.cam_pos = cam_pos



@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, sim_config: Any, episode: Episode) -> Any:
        merged_config = merge_sim_episode_config(sim_config, episode)
        return merged_config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
