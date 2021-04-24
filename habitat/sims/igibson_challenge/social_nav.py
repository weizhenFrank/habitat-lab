# import collections
from typing import Union

# from habitat.core.dataset import Dataset
# from habitat.core.embodied_task import Action, EmbodiedTask, Measure
# from habitat.core.simulator import ActionSpaceConfiguration, Sensor, Simulator
# from habitat.core.utils import Singleton

from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.config.default import Config

from habitat.utils.geometry_utils import (
    get_heading_error,
    quat_to_rad,
)
from habitat.tasks.utils import cartesian_to_polar

import habitat_sim

import random
import quaternion
import magnum as mn
import math
import numpy as np


@registry.register_simulator(name="iGibsonSocialNav")
class iGibsonSocialNav(HabitatSim):
    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        obj_templates_mgr = self.get_object_template_manager()
        self.people_template_ids = obj_templates_mgr.load_configs(
            "/coc/testnvme/nyokoyama3/flash_datasets/igibson_challenge/person_meshes"
        )

    def reset(self) -> Observations:
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        agent_position = self.get_agent_state().position

        # Remove humans
        for id_ in self.get_existing_object_ids():
            self.remove_object(id_)
        self.people = []

        # Spawn humans
        num_people = random.choice([2,3,4])
        min_path_dist = 8
        max_level = 0.6
        obj_templates_mgr = self.get_object_template_manager()
        agent_y = self.get_agent_state(0).position[1]
        for _ in range(num_people):
            person_template_id = random.choice(self.people_template_ids)
            person_id = self.add_object(person_template_id)

            valid_walk = False
            while not valid_walk:
                start = np.array(self.sample_navigable_point())
                goal = np.array(self.sample_navigable_point())
                distance = np.sqrt(
                    (start[0]-goal[0])**2
                    +(start[2]-goal[2])**2
                )
                valid_distance = distance > min_path_dist
                valid_level = (
                    abs(start[1]-agent_position[1]) < max_level
                    and abs(goal[1]-agent_position[1]) < max_level
                )
                valid_walk = valid_distance and valid_level

            start[1] += 0.9 # to get feet on ground, else they sink into floor
            goal[1] += 0.9
            heading = np.random.rand()*2*np.pi-np.pi
            rotation = np.quaternion(np.cos(heading),0,np.sin(heading),0)
            rotation = np.normalized(rotation)
            rotation = mn.Quaternion(
                rotation.imag, rotation.real
            )
            self.set_translation(start, person_id)
            self.set_rotation(rotation, person_id)
            self.set_object_motion_type(
                habitat_sim.physics.MotionType.KINEMATIC,
                person_id
            )
            spf = ShortestPathFollowerv2(sim=self, object_id=person_id)
            spf.get_waypoints(start, goal)
            self.people.append(spf)

        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(sim_obs)

MAX_ANG = np.deg2rad(10)
MAX_LIN = 0.25

class ShortestPathFollowerv2:
    def __init__(
        self,
        sim,
        object_id
    ):
        self._sim = sim
        self.object_id = object_id

        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local    = True
        self.vel_control.ang_vel_is_local    = True

    def get_waypoints(self, start, goal):
        sp = habitat_sim.nav.ShortestPath()
        sp.requested_start = start
        sp.requested_end   = goal
        self._sim.pathfinder.find_path(sp)
        self.waypoints = list(sp.points)+list(sp.points)[::-1][1:-1]
        self.next_waypoint_idx = 1
        self.done_turning = False
        self.current_position = start

        return sp.points

    def step(self, time_step=1):
        waypoint_idx = self.next_waypoint_idx % len(self.waypoints)
        waypoint = np.array(self.waypoints[waypoint_idx])

        translation = self._sim.get_translation(self.object_id)
        mn_quat     = self._sim.get_rotation(self.object_id)

        # Face the next waypoint if we aren't already facing it
        if not self.done_turning:
            # Get current global heading
            heading = np.quaternion(mn_quat.scalar, *mn_quat.vector)
            heading = -quat_to_rad(heading)+np.pi/2

            # Get heading necessary to face next waypoint
            theta = math.atan2(
                waypoint[2]-translation[2], waypoint[0]-translation[0]
            )


            theta_diff = get_heading_error(heading, theta)
            direction = 1 if theta_diff < 0 else -1

            # If next turn would normally overshoot, turn just the right amount
            if MAX_ANG*time_step*1.2 >= abs(theta_diff):
                angular_velocity = -theta_diff / time_step
                self.done_turning = True
            else:
                angular_velocity = MAX_ANG*direction

            self.vel_control.linear_velocity = np.zeros(3)
            self.vel_control.angular_velocity = np.array([
                0.0, angular_velocity, 0.0
            ])

        # Move towards the next waypoint
        else:
            # If next move would normally overshoot, move just the right amount
            distance = np.sqrt(
                (translation[0]-waypoint[0])**2+(translation[2]-waypoint[2])**2
            )
            if MAX_LIN*time_step*1.2 >= distance:
                linear_velocity = distance / time_step
                self.done_turning = False
                self.next_waypoint_idx += 1
            else:
                linear_velocity = MAX_LIN

            self.vel_control.angular_velocity = np.zeros(3)
            self.vel_control.linear_velocity = np.array([
                0.0, 0.0, linear_velocity
            ])

        rigid_state = habitat_sim.bindings.RigidState(
            mn_quat, 
            translation
        )
        rigid_state = self.vel_control.integrate_transform(
            time_step, rigid_state
        )

        self._sim.set_translation(rigid_state.translation, self.object_id)
        self._sim.set_rotation(rigid_state.rotation, self.object_id)
        self.current_position = rigid_state.translation

