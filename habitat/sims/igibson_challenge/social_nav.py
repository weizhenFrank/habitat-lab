# import collections
# from typing import Any, Callable, DefaultDict, Optional, Type

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

import habitat_sim

import random
import quaternion
import magnum as mn
import numpy as np

'''
step function needs to move the people along their paths

'''

@registry.register_simulator(name="iGibsonSocialNav")
class iGibsonSocialNav(HabitatSim):
    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        self.people_template_ids = self.get_object_template_manager().load_configs(
            "/coc/testnvme/nyokoyama3/flash_datasets/igibson_challenge/person_meshes"
        )

    def reset(self) -> Observations:
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()


        # Remove humans
        for id_ in self.get_existing_object_ids():
            self.remove_object(id_)

        # Spawn 20 humans
        obj_templates_mgr = self.get_object_template_manager()
        agent_y = self.get_agent_state(0).position[1]
        for _ in range(20):
            person_template_id = random.choice(self.people_template_ids)
            person_id = self.add_object(person_template_id)
            translation = None
            while (
                translation is None
                or abs(translation[1]-agent_y) > 1.0
            ):
                translation = self.sample_navigable_point()
                translation[1] += 0.9 # to get feet on ground, else they sink
            heading = np.random.rand()*2*np.pi-np.pi
            rotation = np.quaternion(np.cos(heading),0,np.sin(heading),0)
            rotation = np.normalized(rotation)
            rotation = mn.Quaternion(
                rotation.imag, rotation.real
            )
            self.set_translation(translation, person_id)
            self.set_rotation(rotation, person_id)
            self.set_object_motion_type(
                habitat_sim.physics.MotionType.KINEMATIC,
                person_id
            )

        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(sim_obs)