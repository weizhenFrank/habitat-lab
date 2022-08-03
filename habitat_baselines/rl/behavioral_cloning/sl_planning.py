import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from skimage.draw import disk
from torch import nn as nn
from torch.utils.data import DataLoader

MODEL_DIR = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_weights"
IMG_DIR = (
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_planning_1157_imgs"
)
MAP_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/context_maps.npy"
GOAL_PTH = (
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/context_goals.npy"
)
WPT_MAP_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/context_waypoint_maps.npy"
WPT_PTH = (
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/context_waypoints.npy"
)

EVAL_MAP_PTH = (
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/eval_context_maps.npy"
)
EVAL_GOAL_PTH = (
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/eval_context_goals.npy"
)
EVAL_WPT_MAP_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/eval_context_waypoint_maps.npy"
EVAL_WPT_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/eval_context_waypoints.npy"


class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        in_channels = 2
        cnn_out_dim = int((input_shape[0] // 16) * (input_shape[1] // 16))

        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * cnn_out_dim, hidden_size),
        )
        self.mlp = nn.Sequential(
            nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3)
        )

    def forward(self, x):
        feats = self.encoder(x)
        pred_out = self.mlp(feats)
        return pred_out


class Planner:
    def __init__(self):
        self.device = "cuda"
        self.input_size = 256
        self.radius = 5
        self.hidden_size = 512
        self.input_shape = (self.input_size, self.input_size)
        self.mpp = 0.1
        self.setup_networks(self.input_shape)

        LR = 2.5e-4
        self.optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=LR,
        )
        self.batch_length = 8

    def load_data(self, map_pth, goal_pth, wpt_pth, shuffle=True):
        input = self.setup_input(map_pth, goal_pth)
        output = self.setup_output(wpt_pth)
        dataset = tuple(zip(input, output))

        return DataLoader(
            dataset, batch_size=self.batch_length, shuffle=shuffle
        )

    def setup_input(self, map_pth, goal_pth):
        context_maps = np.load(map_pth).astype(np.float32)
        context_goals = np.load(goal_pth).astype(np.float32)

        return np.concatenate(
            [
                np.expand_dims(context_maps, axis=1),
                np.expand_dims(context_goals, axis=1),
            ],
            axis=1,
        )

    def setup_output(self, wpt_pth):
        context_waypoints = np.load(wpt_pth).astype(np.float32)
        return np.array(
            [
                np.exp(context_waypoints[:, 0]),
                np.sin(context_waypoints[:, 1]),
                np.cos(context_waypoints[:, 1]),
            ]
        ).T

    def setup_networks(self, input_shape):
        self.model = Encoder(input_shape, self.hidden_size).to(self.device)

    def compute_r_theta(self, goal_coord):
        origin = np.array([self.input_shape[0] // 2, self.input_shape[0] // 2])

        r = np.linalg.norm(origin - goal_coord)
        theta = np.arctan2(
            goal_coord[1] - origin[1], goal_coord[0] - origin[0]
        )

        return torch.tensor(
            [r, np.sin(theta), np.cos(theta)],
            dtype=torch.float,
            device=self.device,
        )

    def train(self):
        batch_num = 0.0
        action_loss = 0.0
        self.train_dataloader = self.load_data(MAP_PTH, GOAL_PTH, WPT_PTH)
        for i in range(20000):
            input_map, label_wpt_vec = iter(self.train_dataloader).next()
            pred_out = self.model(input_map.to(self.device))
            action_loss = F.mse_loss(pred_out, label_wpt_vec.to(self.device))

            # action_loss += loss
            self.optimizer.zero_grad()
            # action_loss /= float(self.batch_length)
            action_loss.backward()
            self.optimizer.step()
            print(
                "pred: ",
                pred_out,
                "label: ",
                label_wpt_vec,
            )
            batch_num += 1
            print(f"#{batch_num}" f"  action_loss: {action_loss.item():.4f}")
            action_loss = 0

        torch.save(
            self.model.state_dict(),
            os.path.join(MODEL_DIR, "planning_model.pth"),
        )

    def r_theta_to_coord(self, r_theta):
        r, sin_theta, cos_theta = r_theta[0]
        theta = np.arctan2(sin_theta, cos_theta)
        x = (r / self.mpp) * np.cos(theta)
        y = (r / self.mpp) * np.sin(theta)
        mid = self.input_shape[0] // 2
        row, col = np.clip(
            int(mid - x), 0 + self.radius, self.input_size - self.radius
        ), np.clip(
            int(mid - y), 0 + self.radius, self.input_size - self.radius
        )
        return row, col
        # return x, y

    def eval(self, model_pth):
        self.model.load_state_dict(torch.load(os.path.abspath(model_pth)))
        self.model.eval()
        self.batch_length = 1
        EVAL_MAP_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/1157_data/context_maps.npy"
        EVAL_GOAL_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/1157_data/context_goals.npy"
        EVAL_WPT_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/1157_data/context_waypoints.npy"

        self.val_dataloader = self.load_data(
            EVAL_MAP_PTH, EVAL_GOAL_PTH, EVAL_WPT_PTH, shuffle=True
        )
        action_loss = 0
        num_iter = 200
        with torch.no_grad():
            for i in range(num_iter):
                input_map, label_wpt_vec = iter(self.val_dataloader).next()
                print("label_wpt_vec: ", label_wpt_vec)
                pred_out = self.model(input_map.to(self.device))
                cat_imgs = []
                # input map
                cat_imgs.append(input_map[0, 0, :, :].cpu().numpy() * 255.0)
                cat_imgs.append(np.ones((self.input_size, 1)) * 255.0)

                # input curr position and goal
                cat_imgs.append(input_map[0, 1, :, :].cpu().numpy() * 255.0)
                cat_imgs.append(np.ones((self.input_size, 1)) * 255.0)

                # # predicted waypoint
                pred_img = np.zeros(self.input_shape)
                x, y = self.r_theta_to_coord(pred_out.detach().cpu().numpy())
                rr, cc = disk((x, y), self.radius)
                pred_img[rr, cc] = 1.0
                # cat_imgs.append(pred_img * 255.0)
                # cat_imgs.append(np.ones((self.input_size, 1)))

                # overlay
                overlay = input_map[0, 0, :, :].cpu().numpy().copy()
                overlay[input_map[0, 1, :, :] == 1] = 0.3
                overlay[pred_img == 1] = 0.7
                cat_imgs.append(overlay * 255.0)
                cat_imgs.append(np.ones((self.input_size, 1)) * 255.0)

                # overlay gt
                overlay_gt = input_map[0, 0, :, :].cpu().numpy().copy()
                overlay_gt[input_map[0, 1, :, :] == 1] = 0.3

                # gt waypoint
                gt_img = np.zeros(self.input_shape)
                x, y = self.r_theta_to_coord(
                    label_wpt_vec.detach().cpu().numpy()
                )
                print("label: ", label_wpt_vec, "x: ", x, "y: ", y)

                rr, cc = disk((x, y), self.radius)
                gt_img[rr, cc] = 1.0

                overlay_gt[gt_img == 1] = 0.7
                cat_imgs.append(overlay_gt * 255.0)

                action_loss += F.mse_loss(
                    pred_out, label_wpt_vec.to(self.device)
                )
                print(
                    "pred: ",
                    pred_out,
                    "label: ",
                    label_wpt_vec,
                    "loss: ",
                    action_loss,
                )

                cat_img = np.concatenate(cat_imgs, axis=1)
                cv2.imwrite(os.path.join(IMG_DIR, f"cat_{i}.png"), cat_img)
            print("action loss: ", action_loss / num_iter)


if __name__ == "__main__":
    P = Planner()
    if sys.argv[1] == "train":
        print("training!")
        P.train()
    elif sys.argv[1] == "eval":
        print("evaluating!")
        P.eval(sys.argv[2])
    else:
        print("please specify train or eval")
