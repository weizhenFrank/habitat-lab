import glob
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ppo.policy import *
from pytorch_grad_cam import (AblationCAM, EigenCAM, FullGrad, GradCAM,
                              GradCAMPlusPlus, HiResCAM, ScoreCAM, XGradCAM)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# WEIGHT = "spot_depth_context_resnet18_full_map_prevact_sincos32_log_rot_256_0.5_stacked_no_sc_bp_0.3_norm_input_no_avgpool_sd_1"
WEIGHT = "spot_depth_context_resnet18_map_prevact_sincos32_log_rot_256_0.5_robot_scale_0.1_stacked_no_sc_sd_1"
WEIGHTS_PTH = f"/coc/testnvme/jtruong33/results/outdoor_nav_results/{WEIGHT}/checkpoints/ckpt.61.pth"


MAPS_DIR = "/coc/testnvme/jtruong33/google_nav/habitat-lab/maps_npy"


class ContextNavPolicy:
    def __init__(self, checkpoint_path, device):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # for k, v in checkpoint["state_dict"].items():
        #     print(k)
        config = checkpoint["config"]
        """ Disable observation transforms for real world experiments """
        config.defrost()
        config.RL.POLICY["normalize_visual_inputs"] = False
        config.freeze()
        self.map_shape = 256
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
                "context_map": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(self.map_shape, self.map_shape, 3),
                    dtype=np.float32,
                ),
            }
        )
        # Linear and angular velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (2,))
        action_space.n = 2
        self.policy = PointNavContextPolicy.from_config(
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
        # target_layers = [self.policy.net.context_encoder.layer4[-1]]
        # self.cam = cam = GradCAM(
        #     model=self.policy.net.context_encoder,
        #     target_layers=target_layers,
        #     use_cuda=True,
        # )
        self.conv_layers, self.model_weights = self.get_conv_layers()
        self.ctr = 0

    def get_conv_layers(self):
        model_weights = []
        conv_layers = []
        l = [
            module
            for module in self.policy.net.context_encoder.modules()
            if not isinstance(module, nn.Sequential)
        ]
        for ll in l:
            if isinstance(ll, nn.Conv2d):
                conv_layers.append(ll)
                model_weights.append(ll.weight)
        print("len(conv_layers): ", len(conv_layers))
        return conv_layers, model_weights

    def act(self, obs_np):
        with torch.no_grad():
            image = torch.from_numpy(obs_np).to(self.device)
            outputs = []
            names = []
            for layer in self.conv_layers[0:]:
                try:
                    image = layer(image)
                    outputs.append(image)
                    names.append(str(layer))
                except:
                    pass
            processed = []
            for feature_map in outputs:
                feature_map = feature_map.squeeze(0)
                gray_scale = torch.sum(feature_map, 0)
                gray_scale = gray_scale / feature_map.shape[0]
                processed.append(gray_scale.data.cpu().numpy())

            fig = plt.figure(figsize=(30, 50))
            for i in range(len(processed)):
                a = fig.add_subplot(5, 4, i + 1)
                imgplot = plt.imshow(processed[i])
                a.axis("off")
                a.set_title(names[i].split("(")[0], fontsize=30)
            plt.savefig(
                str(f"feature_maps/feature_maps_{self.ctr}.jpg"), bbox_inches="tight"
            )
            plt.close(fig)

            print("saved")
            self.ctr += 1
        return


if __name__ == "__main__":
    nav_policy = ContextNavPolicy(
        WEIGHTS_PTH,
        device="cuda",
    )
    for file in sorted(glob.glob(os.path.join(MAPS_DIR, "*.npy"))):
        nav_policy.act(np.load(file))
