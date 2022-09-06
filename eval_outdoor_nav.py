import glob
import os
import time

import cv2
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.utils.common import batch_obs
from torch.distributions import Normal

from indoor2outdoornav.outdoor_policy import OutdoorPolicy
from indoor2outdoornav.outdoor_ppo import OutdoorDDPPO, OutdoorPPO

WEIGHTS_PTH = "/coc/pskynet3/jtruong33/develop/flash_results/outdoor_nav_results/spot_depth_simple_cnn_cutout_nhy_2hz_ny_rand_pitch_odn_8env_8gpu_v3_sd_1/checkpoints/ckpt.3.pth"
REAL_IMG_DIR = (
    "/coc/testnvme/jtruong33/data/outdoor_imgs/outdoor_inspection_route/filtered"
)
SIM_IMG_DIR = "/coc/testnvme/jtruong33/data/outdoor_imgs/sim_hm3d_eval_imgs"
# OUT_IMG_DIR = "/coc/testnvme/jtruong33/google_nav/habitat-lab/bottleneck_imgs"
OUT_IMG_DIR = "/coc/testnvme/jtruong33/google_nav/habitat-lab/bottleneck_imgs_sim"


class OutdoorNavPolicy:
    def __init__(self, checkpoint_path, device):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for k, v in checkpoint["state_dict"].items():
            print(k)
        config = checkpoint["config"]
        """ Disable observation transforms for real world experiments """
        # config.defrost()
        # config.freeze()
        observation_space = SpaceDict(
            {
                "spot_left_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "spot_right_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        # Linear and angular velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (2,))
        action_space.n = 2
        self.policy = OutdoorPolicy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        print("Actor-critic architecture:", self.policy)
        # Move it to the device
        self.policy.to(self.device)
        # Load trained weights into the policy
        self.policy.load_state_dict(
            {k[len("actor_critic.") :]: v for k, v in checkpoint["state_dict"].items()},
            strict=False,
        )
        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config
        self.num_actions = action_space.shape[0]
        self.reset_ran = False
        self.count = 0

    def reset(self):
        self.reset_ran = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def _sample_vis_feats(self, mu_std, deterministic=False):
        dist = Normal(*mu_std)
        if deterministic:
            return dist.mean
        else:
            return dist.rsample()

    def act(self, obs_np):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        with torch.no_grad():
            obs = torch.from_numpy(
                np.expand_dims(obs_np, axis=0)[:, :, :, :1] / 255.0
            ).to(device=self.device, dtype=torch.float32)
            real_obs = {"depth": obs}
            width = obs.shape[2]
            right_depth, left_depth = torch.split(obs, int(width / 2), 2)
            obs = {
                "spot_right_depth": right_depth,
                "spot_left_depth": left_depth,
            }

            visual_features = self.policy.net.visual_encoder(obs)
            sampled_vis_feats = self._sample_vis_feats(visual_features)

            sim_pred = self.policy.net.sim_visual_decoder(sampled_vis_feats)
            sim_pred = sim_pred.detach().cpu().numpy().squeeze() * 255

            reality_visual_features = self.policy.net.reality_visual_encoder(real_obs)
            sampled_real_vis_feats = self._sample_vis_feats(reality_visual_features)

            reality_pred = self.policy.net.reality_visual_decoder(
                sampled_real_vis_feats
            )
            reality_pred = reality_pred.detach().cpu().numpy().squeeze() * 255

            sim_pred_label = self.policy.net.discriminator(sampled_vis_feats)
            real_pred_label = self.policy.net.discriminator(sampled_real_vis_feats)
            print("sim pred: ", sim_pred_label, " real pred: ", real_pred_label)

        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        all_real_pth = os.path.join(
            f"{OUT_IMG_DIR}/real_depth_pred_{self.count:06}.png"
        )
        all_sim_pth = os.path.join(f"{OUT_IMG_DIR}/sim_depth_pred_{self.count:06}.png")
        cv2.imwrite(all_sim_pth, sim_pred)
        # cv2.imwrite(all_real_pth, reality_pred)
        self.count += 1
        return


if __name__ == "__main__":
    nav_policy = OutdoorNavPolicy(
        WEIGHTS_PTH,
        device="cuda",
    )
    nav_policy.reset()

    # for file in sorted(glob.glob(os.path.join(REAL_IMG_DIR, "*.png"))):
    for file in sorted(glob.glob(os.path.join(SIM_IMG_DIR, "*.png"))):
        print(file)
        img = cv2.imread(file)
        nav_policy.act(img)
