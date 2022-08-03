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

MODEL_DIR = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_weights"
IMG_DIR = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_imgs"


class ImgtoCoord:
    def __init__(self):
        self.device = "cuda"
        self.input_size = 256
        self.input_shape = (self.input_size, self.input_size)
        self.mpp = 0.5
        self.setup_networks(self.input_shape)

        LR = 2.5e-4
        self.optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=LR,
        )
        self.batch_length = 8
        self.X = self.get_truncated_normal(mean=1.0, sd=0.1, low=0, upp=1)

    def setup_networks(self, input_shape):
        in_channels = 1
        cnn_out_dim = int((input_shape[0] // 16) * (input_shape[1] // 16))

        self.model = nn.Sequential(
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
            nn.Linear(32 * cnn_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        ).to(device=self.device)

    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
        )

    def get_rand_map_label(self):
        blank = np.zeros(
            (self.input_size, self.input_size, 1), dtype=np.float32
        )
        radius = 5
        row = np.random.randint(0 + radius, self.input_size - radius)
        col = np.random.randint(0 + radius, self.input_size - radius)
        rr, cc = disk((row, col), radius)
        b = self.X.rvs(blank[rr, cc, :].shape)
        blank[rr, cc, :] = b
        blank = np.expand_dims(blank, axis=0)
        label_coord = np.array([row, col])
        label_r_theta = self.compute_r_theta(label_coord)
        return torch.tensor(blank, device=self.device), label_r_theta

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

    def r_theta_to_coord(self, r_theta):
        r, sin_theta, cos_theta = r_theta[0]
        theta = np.arctan2(sin_theta, cos_theta)
        y = r * np.cos(theta)
        x = r * np.sin(theta)

        origin = np.array([self.input_shape[0] // 2, self.input_shape[0] // 2])
        return origin[0] + y, origin[0] + x

    def train(self):
        batch_num = 0.0
        action_loss = 0.0
        for iteration in range(20000):
            rand_img, label = self.get_rand_map_label()
            rand_img = rand_img.permute(0, 3, 1, 2)
            pred_out = self.model(rand_img)
            loss = F.mse_loss(pred_out, label)
            action_loss += loss
            if iteration % self.batch_length == 0:
                self.optimizer.zero_grad()
                action_loss /= float(self.batch_length)
                action_loss.backward()
                self.optimizer.step()
                batch_num += 1
                print(
                    f"#{batch_num}" f"  action_loss: {action_loss.item():.4f}"
                )
                action_loss = 0

        torch.save(
            self.model.state_dict(),
            os.path.join(MODEL_DIR, "model.pth"),
        )

    def eval(self, model_pth):
        self.model.load_state_dict(torch.load(os.path.abspath(model_pth)))
        self.model.eval()
        action_loss = 0
        with torch.no_grad():
            for i in range(100):
                rand_img, label = self.get_rand_map_label()
                rand_img = rand_img.permute(0, 3, 1, 2)
                pred_out = self.model(rand_img)
                ## get input img
                cv_img = rand_img[0, 0, :, :].cpu().numpy() * 255.0
                cat_imgs = []
                cat_imgs.append(cv_img)
                ## get pred img
                pred_img = np.zeros(self.input_shape)
                x, y = self.r_theta_to_coord(pred_out.detach().cpu().numpy())
                rr, cc = disk((x, y), 2)
                b = self.X.rvs(pred_img[rr, cc].shape)
                pred_img[rr, cc] = b
                cat_imgs.append(pred_img * 255.0)
                cat_img = np.concatenate(cat_imgs, axis=1)
                cv2.imwrite(os.path.join(IMG_DIR, f"cat_{i}.png"), cat_img)
                loss = F.mse_loss(pred_out, label)
                print("LABEL: ", label, "PRED: ", pred_out, "LOSS: ", loss)
                action_loss += loss
            print("action loss: ", action_loss / 20)


if __name__ == "__main__":
    IC = ImgtoCoord()
    if sys.argv[1] == "train":
        print("training!")
        IC.train()
    elif sys.argv[1] == "eval":
        print("evaluating!")
        IC.eval(sys.argv[2])
