from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    PointNavResNetPolicy,
)
from habitat_baselines.utils.common import batch_obs
from habitat.core.spaces import ActionSpace

import cv2
from gym import spaces
from gym.spaces import Dict as SpaceDict
import torch
import quaternion as qt
import magnum as mn
import numpy as np

from habitat_sim.bindings import RigidState
from habitat_sim.physics import VelocityControl
from habitat_sim.utils.common import quat_rotate_vector


def heading_to_quaternion(heading):
    quat = qt.as_float_array(
        qt.from_euler_angles([heading + np.pi / 2, 0, 0, 0])
    )
    quat = [quat[1], -quat[3], quat[2], quat[0]]
    quat = qt.quaternion(*quat)
    return mn.Quaternion(quat.imag, quat.real)


def quat_to_rad(quat):
    heading_vector = quat_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = np.arctan2(heading_vector[0], -heading_vector[2])
    return phi - np.pi / 2


# Turn numpy observations into torch tensors for consumption by policy
def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class SpotNavPolicy:
    def __init__(self, checkpoint_path, device):
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        config = checkpoint["config"]

        """ Disable observation transforms for real world experiments """
        config.defrost()
        config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
        config.freeze()

        """ Define observation space (needed for policy instantiation) """
        # Extract config used spot domain randomization
        spot_dr_config = config.RL.POLICY.OBS_TRANSFORMS.SPOT_DR

        # Use DR observation transform dimensions, since it's the final
        # transform right before data is sent to policy
        self.depth_height = spot_dr_config.HEIGHT
        self.depth_width = spot_dr_config.WIDTH

        observation_space = SpaceDict(
            {
                "depth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.depth_height, self.depth_width, 1),
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        """ Define action space (needed for policy instantiation) """
        vel_config = config.TASK_CONFIG.TASK.ACTIONS.VELOCITY_CONTROL
        print("VELOCITY CONFIGURATION EXTRACTED FROM CHECKPOINT:")
        print(vel_config)

        # Action space parameters
        self.min_lin_vel, self.max_lin_vel = vel_config.LIN_VEL_RANGE
        self.min_ang_vel, self.max_ang_vel = vel_config.ANG_VEL_RANGE
        self.min_abs_lin_speed = vel_config.MIN_ABS_LIN_SPEED
        self.min_abs_ang_speed = vel_config.MIN_ABS_ANG_SPEED
        self.must_call_stop = vel_config.MUST_CALL_STOP
        self.time_step = vel_config.TIME_STEP

        # Horizontal velocity
        self.min_hor_vel, self.max_hor_vel = vel_config.HOR_VEL_RANGE
        self.has_hor_vel = self.min_hor_vel != 0.0 and self.max_hor_vel != 0.0
        self.min_abs_hor_speed = vel_config.MIN_ABS_HOR_SPEED

        action_dict = {
            "linear_velocity": spaces.Box(
                low=np.array([self.min_lin_vel]),
                high=np.array([self.max_lin_vel]),
                dtype=np.float32,
            ),
            "angular_velocity": spaces.Box(
                low=np.array([self.min_ang_vel]),
                high=np.array([self.max_ang_vel]),
                dtype=np.float32,
            ),
        }

        if self.has_hor_vel:
            action_dict["horizontal_velocity"] = spaces.Box(
                low=np.array([self.min_hor_vel]),
                high=np.array([self.max_hor_vel]),
                dtype=np.float32,
            )

        action_space = ActionSpace(action_dict)

        # Now we can finally instantiate the policy
        self.policy = PointNavResNetPolicy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )

        # Move it to the device
        self.policy.to(self.device)

        # Load trained weights into the policy
        self.policy.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        )

        # Velocity control object needed to turn velocities into next waypoint
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(
            1, 1, dtype=torch.bool, device=self.device
        )

        if self.has_hor_vel:
            self.prev_actions = torch.zeros(1, 3, device=self.device)
        else:
            self.prev_actions = torch.zeros(1, 2, device=self.device)

    # TODO: right now this only expects/supports depth, not rgb.
    def act(self, depth, rho, theta):
        """

        :param depth: cv2 mono-channel image, where dtype is float
        :param pointgoal_with_gps_compass:
        :return: three floats corresponding to linear, angular, and horizontal velocity
        """

        # Resize depth image
        depth_resized = cv2.resize(
            depth,
            (self.depth_width, self.depth_height),
            interpolation=cv2.INTER_AREA,
        )

        # Add channel dimension if it's missing
        if len(depth_resized.shape) < 3:
            depth_resized = np.expand_dims(depth_resized, axis=-1)

        # Ensure type of float, not double
        depth_resized = np.float32(depth_resized)

        # Collate inputs into the proper observation batch format
        observations = {
            "depth": depth_resized,
            "pointgoal_with_gps_compass": (rho, theta),
        }
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )
        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(
            1, 1, dtype=torch.bool, device=self.device
        )

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        # Convert from [-1, 1] to [0, 1] range
        actions = (actions + 1.0) / 2.0

        # Scale each action
        lin_vel, ang_vel = actions[:2]
        lin_vel = self.min_lin_vel + lin_vel * (
            self.max_lin_vel - self.min_lin_vel
        )
        ang_vel = self.min_ang_vel + ang_vel * (
            self.max_ang_vel - self.min_ang_vel
        )
        ang_vel = np.deg2rad(ang_vel)

        if self.has_hor_vel:
            hor_vel = actions[2]
            hor_vel = self.min_hor_vel + hor_vel * (
                self.max_hor_vel - self.min_hor_vel
            )

            return lin_vel, ang_vel, hor_vel

        return lin_vel, ang_vel, 0.0

    def pointgoal(self, curr_x, curr_y, curr_yaw, goal_x, goal_y):
        """
        Get relative distance (rho) and heading (theta) to given goal from current
        position and heading
        """
        curr_coordinates = np.array([curr_x, curr_y])
        goal = np.array([goal_x, goal_y])
        rho = np.linalg.norm(curr_coordinates - goal)
        theta = np.arctan2(goal_y - curr_y, goal_x - curr_x) - curr_yaw

        if theta >= np.pi:
            theta -= 2 * np.pi
        elif theta < -np.pi:
            theta += 2 * np.pi

        return rho, theta

    def velocites_to_waypoint(
        self, current_position, current_heading, lin_vel, ang_vel, hor_vel
    ):
        # z is 2nd dim in Habitat, x is 3rd AND FLIPPED!
        self.vel_control.linear_velocity = np.array([hor_vel, 0.0, -lin_vel])
        self.vel_control.angular_velocity = np.array([0.0, ang_vel, 0.0])

        current_quaternion = heading_to_quaternion(current_heading)
        current_rigid_state = RigidState(
            current_quaternion,
            mn.Vector3(current_position[0], 0.0, current_position[1]),
        )

        # manually integrate the rigid state
        next_rigid_state = self.vel_control.integrate_transform(
            self.time_step, current_rigid_state
        )

        # Get new position, convert from quaternion to euler angle
        next_position = np.array(
            [next_rigid_state.translation[0], next_rigid_state.translation[2]],
            dtype=np.float32,
        )
        next_heading = quat_to_rad(
            qt.quaternion(
                next_rigid_state.rotation.scalar,
                *next_rigid_state.rotation.vector,
            )
        )

        return next_position, next_heading


if __name__ == "__main__":
    snp = SpotNavPolicy(
        "/private/home/naokiyokoyama/spot_nav/exp/maiden/checkpoints/maiden/ckpt.87.pth",
        device="cuda",
    )
    snp.reset()

    # Depth image
    depth_img = (
        np.float32(
            cv2.imread(
                "/private/home/naokiyokoyama/spot_nav/test_depth_images/depth_image_1.png",
                cv2.IMREAD_GRAYSCALE,
            )
        )
        / 255.0
    )

    # Relative distance and heading to goal. Rho must be positive.
    rho, theta = np.array([1.0, 0.0], dtype=np.float32)

    # Where the robot current is in global 2D map. Used to find next waypoint.
    current_position = np.array([0.0, 0.0], dtype=np.float32)
    current_heading = 0.0

    lin_vel, ang_vel, hor_vel = snp.act(depth_img, rho, theta)

    lin_vel, ang_vel, hor_vel = 1.0, np.pi/2, 0.0
    next_position, next_heading = snp.velocites_to_waypoint(
        current_position=current_position,
        current_heading=current_heading,
        lin_vel=lin_vel,
        ang_vel=ang_vel,
        hor_vel=hor_vel,
    )


    print("Velocities inferred by policy [lin, ang (deg), hor]:")
    print(
        " ".join(
            [str(round(i, 3)) for i in [lin_vel, np.rad2deg(ang_vel), hor_vel]]
        )
    )
    print(
        f"Next waypoint from {(*current_position,)} using "
        "above velocities [x, y, theta (deg)]:"
    )
    print(
        " ".join(
            [
                str(round(i, 3))
                for i in [*next_position, np.rad2deg(next_heading)]
            ]
        )
    )
